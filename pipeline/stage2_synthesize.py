"""
pipeline/stage2_synthesize.py
──────────────────────────────
Stage 2 — TTS Synthesis.

Supported engines
─────────────────
  edge-tts    Microsoft Edge neural voices (ar-EG-SalmaNeural, ar-EG-ShakirNeural)
              Free, async, no GPU required.

  egtts       OmarSamir/EGTTS-V0.1 — XTTS-v2 fine-tune for Egyptian Arabic.
              Requires a reference WAV for voice cloning.

  niletts     KickItLikeShika/NileTTS-XTTS — XTTS-v2 fine-tune trained on 38 h
              of Egyptian Arabic speech (medical, sales, general conversation).
              Requires a reference WAV for voice cloning.

  google-     Google Cloud Text-to-Speech (ar-EG voices).
  vertex      Requires GOOGLE_APPLICATION_CREDENTIALS in environment.

Resumability
────────────
Every job (prompt × engine × voice) is tracked in a SQLite checkpoint DB.
Completed jobs are skipped on re-runs.  Failed jobs are retried up to
`max_retries` times.

Output manifest schema (appended to data/manifests/manifest_<run_id>.jsonl):
  {
    "id":              "p0001_edgetts_SalmaNeural",
    "prompt_id":       "p0001",
    "text":            "إزيك النهارده؟",
    "domain":          "conversation",
    "source":          "seed",
    "engine":          "edge-tts",
    "voice":           "ar-EG-SalmaNeural",
    "audio_path":      "data/audio/p0001_edgetts_SalmaNeural.wav",
    "sample_rate":     24000,
    "duration_sec":    1.234,
    "snr_db":          28.5,
    "silence_ratio":   0.04,
    "quality_flags":   [],
    "quality_passed":  true,
    "review_status":   "pending",
    "review_note":     null,
    "synthesized_at":  "2026-05-15T10:00:00+00:00"
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from utils.checkpointing import CheckpointDB
from utils.quality import quality_check

log = logging.getLogger(__name__)

TARGET_SR = 24_000  # unified output sample rate


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_stem(text: str) -> str:
    """Return a filesystem-safe slug for use in filenames."""
    return re.sub(r"[^\w]", "_", text)[:30]


def _mp3_to_wav(mp3_path: str | Path, wav_path: str | Path, target_sr: int = TARGET_SR) -> None:
    """Convert an MP3 file to a normalised mono WAV using torchaudio."""
    waveform, sr = torchaudio.load(str(mp3_path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(str(wav_path), waveform, target_sr)


# ── Abstract engine interface ─────────────────────────────────────────────────

class TTSEngine(ABC):
    """Base class for all TTS engine adapters."""

    name: str = "base"

    @abstractmethod
    def synthesize(self, text: str, voice: str, output_path: str | Path) -> bool:
        """
        Synthesize *text* to a WAV file at *output_path*.

        Returns True on success, False on failure.
        """

    def voices(self) -> List[str]:
        """Return the list of voice IDs this engine supports."""
        return []

    def is_available(self) -> bool:
        """Return True if this engine can run right now."""
        return True


# ── edge-tts engine ───────────────────────────────────────────────────────────

class EdgeTTSEngine(TTSEngine):
    """
    Microsoft Edge neural TTS via the edge-tts library.
    Free, no API key required.  Voices: ar-EG-SalmaNeural, ar-EG-ShakirNeural.
    """

    name = "edge-tts"

    def __init__(self, cfg: Dict) -> None:
        self._voices: List[str] = cfg.get(
            "voices", ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"]
        )

    def voices(self) -> List[str]:
        return self._voices

    def synthesize(self, text: str, voice: str, output_path: str | Path) -> bool:
        try:
            import edge_tts  # noqa: PLC0415
        except ImportError:
            log.error("edge-tts not installed.  Run: pip install edge-tts")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async def _run() -> None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(tmp_path)
                _mp3_to_wav(tmp_path, output_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        try:
            asyncio.run(_run())
            return True
        except Exception as exc:
            log.warning("edge-tts failed for voice=%s: %s", voice, exc)
            return False


# ── XTTS-based engine (EGTTS-V0.1 and NileTTS) ───────────────────────────────

class XTTSEngine(TTSEngine):
    """
    Coqui XTTS-v2 fine-tune engine.

    Handles both OmarSamir/EGTTS-V0.1 and KickItLikeShika/NileTTS-XTTS.
    The model is downloaded from HuggingFace Hub on first use and cached.
    A reference WAV file is required for speaker conditioning.
    """

    def __init__(self, cfg: Dict) -> None:
        self.name: str = cfg["name"]
        self._model_repo: str = cfg["model_repo"]
        self._reference_audio: str = cfg.get("reference_audio", "")
        self._use_deepspeed: bool = cfg.get("use_deepspeed", False)
        self._temperature: float = cfg.get("temperature", 0.75)
        self._device_str: str = cfg.get("device", "cuda")
        self._model = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None

    def is_available(self) -> bool:
        ref = Path(self._reference_audio)
        if not ref.exists():
            log.warning(
                "[%s] Reference audio not found: %s  "
                "Run: python scripts/download_references.py",
                self.name,
                self._reference_audio,
            )
            return False
        try:
            from TTS.tts.configs.xtts_config import XttsConfig  # noqa: F401
        except ImportError:
            log.warning(
                "[%s] Coqui TTS not installed.  "
                "Run: pip install TTS  (or pip install git+https://github.com/coqui-ai/TTS)",
                self.name,
            )
            return False
        return True

    def _load_model(self) -> None:
        """Download (if needed) and load the XTTS model into GPU memory."""
        if self._model is not None:
            return

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from huggingface_hub import snapshot_download

        log.info("[%s] Downloading / loading model from %s …", self.name, self._model_repo)
        model_dir = snapshot_download(
            repo_id=self._model_repo,
            ignore_patterns=["*.md", "*.txt"],
        )

        device = (
            torch.device("cuda")
            if self._device_str == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        config = XttsConfig()
        config.load_json(str(Path(model_dir) / "config.json"))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=model_dir,
            vocab_path=str(Path(model_dir) / "vocab.json"),
            use_deepspeed=self._use_deepspeed,
        )
        model.to(device)
        model.eval()

        log.info("[%s] Computing speaker latents from %s …", self.name, self._reference_audio)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[self._reference_audio],
            gpt_cond_len=6,
            max_ref_length=30,
            sound_norm_refs=False,
        )

        self._model = model
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding
        log.info("[%s] Model ready on %s.", self.name, device)

    def voices(self) -> List[str]:
        # The voice is determined by the reference audio; we expose a single
        # pseudo-voice named after the model.
        return [self.name]

    def synthesize(self, text: str, voice: str, output_path: str | Path) -> bool:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._load_model()
            out = self._model.inference(
                text=text,
                language="ar",
                gpt_cond_latent=self._gpt_cond_latent,
                speaker_embedding=self._speaker_embedding,
                temperature=self._temperature,
            )
            waveform = torch.tensor(out["wav"]).unsqueeze(0)
            torchaudio.save(str(output_path), waveform, TARGET_SR)
            return True
        except Exception as exc:
            log.warning("[%s] Synthesis failed: %s", self.name, exc)
            return False

    def unload(self) -> None:
        """Free GPU memory when the engine is no longer needed."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── Google Cloud / Vertex TTS engine ─────────────────────────────────────────

class GoogleVertexEngine(TTSEngine):
    """Google Cloud Text-to-Speech (ar-EG voices)."""

    name = "google-vertex"

    def __init__(self, cfg: Dict) -> None:
        self._voice_name: str = cfg.get("voice_name", "ar-EG-Standard-A")
        self._speaking_rate: float = cfg.get("speaking_rate", 1.0)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import texttospeech  # noqa: PLC0415
            self._client = texttospeech.TextToSpeechClient()
        return self._client

    def is_available(self) -> bool:
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds or not Path(creds).exists():
            log.warning(
                "[google-vertex] GOOGLE_APPLICATION_CREDENTIALS not set "
                "or file not found — engine disabled."
            )
            return False
        try:
            from google.cloud import texttospeech  # noqa: F401
        except ImportError:
            log.warning(
                "[google-vertex] google-cloud-texttospeech not installed. "
                "Run: pip install google-cloud-texttospeech"
            )
            return False
        return True

    def voices(self) -> List[str]:
        return [self._voice_name]

    def synthesize(self, text: str, voice: str, output_path: str | Path) -> bool:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from google.cloud import texttospeech

            client = self._get_client()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="ar-EG",
                name=voice,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=TARGET_SR,
                speaking_rate=self._speaking_rate,
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            output_path.write_bytes(response.audio_content)
            return True
        except Exception as exc:
            log.warning("[google-vertex] Synthesis failed: %s", exc)
            return False


# ── Engine factory ────────────────────────────────────────────────────────────

def build_engines(synthesis_cfg: Dict) -> List[TTSEngine]:
    """Instantiate enabled engines from the synthesis config block."""
    engines: List[TTSEngine] = []
    for ecfg in synthesis_cfg.get("engines", []):
        if not ecfg.get("enabled", False):
            continue
        name = ecfg["name"]
        if name == "edge-tts":
            engines.append(EdgeTTSEngine(ecfg))
        elif name in ("egtts", "niletts"):
            engines.append(XTTSEngine(ecfg))
        elif name == "google-vertex":
            engines.append(GoogleVertexEngine(ecfg))
        else:
            log.warning("Unknown engine '%s' — skipped.", name)

    # Filter out engines that can't run in the current environment
    available = [e for e in engines if e.is_available()]
    unavailable = [e.name for e in engines if not e.is_available()]
    if unavailable:
        log.warning("Engines not available (missing deps/files): %s", unavailable)
    return available


# ── Main stage entry point ────────────────────────────────────────────────────

def run_stage2(config: Dict, run_id: str, prompts_manifest: str | Path) -> Path:
    """
    Execute Stage 2 — synthesise audio for every prompt.

    Args:
        config:           Full pipeline config dict.
        run_id:           Pipeline run identifier.
        prompts_manifest: Path to the Stage 1 JSONL output.

    Returns:
        Path to the synthesis manifest JSONL.
    """
    syn_cfg = config["synthesis"]
    quality_cfg = config.get("quality", {})
    audio_dir = Path(syn_cfg.get("audio_dir", "data/audio"))
    manifest_dir = Path(syn_cfg.get("manifest_dir", "data/manifests"))
    db_path = syn_cfg.get("checkpoint_db", "data/synthesis.db")
    max_retries: int = syn_cfg.get("max_retries", 3)
    batch_size: int = syn_cfg.get("batch_size", 10)

    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ──────────────────────────────────────────────────────────
    from pipeline.stage1_generate import load_prompts
    prompts = load_prompts(prompts_manifest)
    log.info("Loaded %d prompts from %s", len(prompts), prompts_manifest)

    # ── Build engines ─────────────────────────────────────────────────────────
    engines = build_engines(syn_cfg)
    if not engines:
        raise RuntimeError("No TTS engines are available.  Check config and dependencies.")

    # ── Build checkpoint DB and register jobs ─────────────────────────────────
    db = CheckpointDB(db_path)
    db.reset_running()  # recover from previous crash

    jobs: List[Dict] = []
    for prompt in prompts:
        for engine in engines:
            for voice in engine.voices():
                voice_slug = _safe_stem(voice)
                job_id = f"{prompt['id']}_{engine.name}_{voice_slug}"
                audio_filename = f"{job_id}.wav"
                jobs.append(
                    {
                        "job_id": job_id,
                        "prompt_id": prompt["id"],
                        "engine": engine.name,
                        "voice": voice,
                        "text": prompt["text"],
                        "domain": prompt.get("domain", ""),
                        "source": prompt.get("source", ""),
                        "audio_path": str(audio_dir / audio_filename),
                    }
                )

    inserted = db.add_jobs(jobs)
    stats = db.get_stats()
    log.info(
        "Checkpoint DB: %d new jobs registered.  Stats: %s",
        inserted,
        stats,
    )

    # ── Engine map for fast lookup ─────────────────────────────────────────────
    engine_map: Dict[str, TTSEngine] = {e.name: e for e in engines}

    # ── Synthesis loop ────────────────────────────────────────────────────────
    from tqdm import tqdm

    def _process_batch(batch: List[Dict]) -> None:
        for job in batch:
            engine = engine_map.get(job["engine"])
            if engine is None:
                continue
            db.mark_running(job["job_id"])
            output_path = Path(job["audio_path"])
            # Skip if audio already exists on disk (extra safety net)
            if output_path.exists() and output_path.stat().st_size > 0:
                db.mark_completed(job["job_id"], str(output_path))
                continue
            success = engine.synthesize(job["text"], job["voice"], output_path)
            if success:
                db.mark_completed(job["job_id"], str(output_path))
            else:
                db.mark_failed(job["job_id"], "synthesis returned False")

    # Process pending jobs engine by engine to avoid GPU thrashing
    for engine in engines:
        pending = db.get_pending(engine=engine.name)
        if not pending:
            log.info("[%s] No pending jobs — skipping.", engine.name)
            continue
        log.info("[%s] Synthesising %d clips …", engine.name, len(pending))
        for i in tqdm(range(0, len(pending), batch_size),
                      desc=f"{engine.name}", unit="batch"):
            _process_batch(pending[i : i + batch_size])

        # Retry failures
        for _ in range(max_retries - 1):
            failed = db.get_failed(max_attempts=max_retries)
            failed_for_engine = [j for j in failed if j["engine"] == engine.name]
            if not failed_for_engine:
                break
            log.info("[%s] Retrying %d failed jobs …", engine.name, len(failed_for_engine))
            for j in tqdm(failed_for_engine, desc=f"{engine.name} retry"):
                engine_map[j["engine"]].synthesize(
                    j["text"], j["voice"], j["audio_path"]
                )
                if Path(j["audio_path"]).exists():
                    db.mark_completed(j["job_id"], j["audio_path"])

        # Free GPU memory after each XTTS engine
        if hasattr(engine, "unload"):
            engine.unload()

    # ── Build synthesis manifest with quality signals ─────────────────────────
    log.info("Running quality checks and building manifest …")
    completed = db.get_completed_audio_paths()  # {job_id: audio_path}
    prompt_map = {p["id"]: p for p in prompts}

    manifest_path = manifest_dir / f"manifest_{run_id}.jsonl"
    written = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for job in tqdm(jobs, desc="Quality checks"):
            job_id = job["job_id"]
            audio_path = completed.get(job_id)
            if audio_path is None:
                continue  # not completed — skip

            qr = quality_check(audio_path, quality_cfg)
            prompt = prompt_map.get(job["prompt_id"], {})

            record = {
                "id": job_id,
                "prompt_id": job["prompt_id"],
                "text": job["text"],
                "domain": job.get("domain", prompt.get("domain", "")),
                "source": job.get("source", prompt.get("source", "")),
                "engine": job["engine"],
                "voice": job["voice"],
                "audio_path": audio_path,
                "sample_rate": TARGET_SR,
                **qr.to_dict(),
                "review_status": "pending",
                "review_note": None,
                "synthesized_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    final_stats = db.get_stats()
    log.info(
        "Stage 2 complete — %d manifest entries written to %s.  DB stats: %s",
        written,
        manifest_path,
        final_stats,
    )
    return manifest_path
