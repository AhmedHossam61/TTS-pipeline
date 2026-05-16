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
import random
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio

from utils.checkpointing import CheckpointDB
from utils.quality import quality_check
from utils.audio_augmentation import add_gaussian_noise

log = logging.getLogger(__name__)

TARGET_SR = 24_000  # unified output sample rate


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_stem(text: str) -> str:
    """Return a filesystem-safe slug for use in filenames."""
    return re.sub(r"[^\w]", "_", text)[:30]


def _configure_ffmpeg_runtime() -> None:
    """Prepend a shared FFmpeg bin directory on Windows if one is installed."""
    if os.name != "nt":
        return

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if any("ffmpeg" in p.lower() and Path(p).exists() for p in path_parts):
        for p in path_parts:
            pl = p.lower()
            if (
                "ffmpeg" in pl
                and Path(p, "avcodec-62.dll").exists()
                and Path(p, "avformat-62.dll").exists()
            ):
                return

    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = []
    if local_appdata:
        candidates.extend(
            sorted(
                local_appdata.glob(
                    "Microsoft/WinGet/Packages/Gyan.FFmpeg.Shared*/*/bin"
                ),
                reverse=True,
            )
        )

    for candidate in candidates:
        if (
            candidate.exists()
            and (candidate / "avcodec-62.dll").exists()
            and (candidate / "avformat-62.dll").exists()
        ):
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            log.info("Prepended shared FFmpeg bin to PATH: %s", candidate)
            return


def _mp3_to_wav(mp3_path: str | Path, wav_path: str | Path, target_sr: int = TARGET_SR) -> None:
    """Convert an MP3 file to a mono WAV using ffmpeg to avoid torchcodec."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not found on PATH")

    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            str(mp3_path),
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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

        # ── Monkey-patch torchaudio.load to use soundfile ──────────────────────
        # torchaudio.load internally calls TorchCodec which requires FFmpeg DLLs
        # that are not available on Windows pre-built wheels.  We replace it with
        # a soundfile-based loader that has no such dependency.
        import soundfile as sf

        def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, backend=None):
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # data shape: (frames, channels) → convert to (channels, frames)
            if channels_first:
                data = data.T
            tensor = torch.from_numpy(data.copy())
            if num_frames > 0:
                tensor = tensor[..., frame_offset:frame_offset + num_frames]
            elif frame_offset > 0:
                tensor = tensor[..., frame_offset:]
            return tensor, sr

        def _sf_save(path, src, sample_rate, channels_first=True, **kwargs):
            arr = src.detach().cpu().numpy()
            # arr shape: (channels, frames) or (frames,)
            if arr.ndim == 2 and channels_first:
                arr = arr.T  # → (frames, channels)
            elif arr.ndim == 2 and not channels_first:
                pass  # already (frames, channels)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]  # mono: flatten
            sf.write(str(path), arr, sample_rate)

        torchaudio.load = _sf_load
        torchaudio.save = _sf_save

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
            # Use soundfile directly — torchaudio.save triggers TorchCodec on Windows
            import soundfile as sf
            arr = out["wav"]
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            sf.write(str(output_path), arr, TARGET_SR)
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

class GeminiTTSEngine(TTSEngine):
    """Gemini-TTS through the Vertex AI Gemini API."""

    name = "gemini-tts"

    def __init__(self, cfg: Dict) -> None:
        self._model_name: str = cfg.get(
            "model_name", "gemini-3.1-flash-tts-preview"
        )
        self._voices: List[str] = cfg.get("voices", ["Charon"])
        self._language_code: str = cfg.get("language_code", "ar-EG")
        self._temperature: float = cfg.get("temperature", 1.0)
        self._project_id: Optional[str] = (
            cfg.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        self._prompt: str = cfg.get(
            "prompt",
            (
                "Read the following text in natural Egyptian Arabic. "
                "Keep the wording unchanged, use a clear conversational tone, "
                "and avoid adding extra words."
            ),
        )
        self._location: str = (
            cfg.get("location")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_REGION")
            or "global"
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # noqa: PLC0415

            self._client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location,
            )
        return self._client

    def voices(self) -> List[str]:
        return self._voices

    def is_available(self) -> bool:
        if not self._project_id:
            log.warning(
                "[gemini-tts] GOOGLE_CLOUD_PROJECT is not set and no project_id "
                "was provided in config."
            )
            return False
        try:
            from google import genai  # noqa: F401
            from google.genai import types  # noqa: F401
        except ImportError:
            log.warning(
                "[gemini-tts] google-genai is not installed. "
                "Run: pip install google-genai"
            )
            return False
        return True

    def synthesize(self, text: str, voice: str, output_path: str | Path) -> bool:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import wave
            from google.genai import types

            client = self._get_client()
            contents = (
                f"{self._prompt}\n\n"
                "Text to synthesize exactly, without rewriting:\n"
                f"{text}"
            )
            response = client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    speech_config=types.SpeechConfig(
                        language_code=self._language_code,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        ),
                    ),
                    temperature=self._temperature,
                ),
            )

            audio_data = response.candidates[0].content.parts[0].inline_data.data
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SR)
                wf.writeframes(audio_data)
            return True
        except Exception as exc:
            log.warning("[gemini-tts] Synthesis failed for voice=%s: %s", voice, exc)
            return False


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
        elif name == "gemini-tts":
            engines.append(GeminiTTSEngine(ecfg))
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

    # Make shared FFmpeg DLLs available to torchcodec/TTS on Windows.
    _configure_ffmpeg_runtime()

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
                job_id = f"{run_id}_{prompt['id']}_{engine.name}_{voice_slug}"
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

    sync_stats = db.sync_jobs(jobs)
    stats = db.get_stats()
    log.info(
        "Checkpoint DB: %d new jobs registered, %d unfinished jobs refreshed, "
        "%d completed changed jobs kept. Stats: %s",
        sync_stats["inserted"],
        sync_stats["updated"],
        sync_stats["kept_completed"],
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
            output_path_str = job.get("audio_path")
            if not output_path_str:
                # Backward compatibility for older DB rows inserted before
                # audio_path was persisted in synthesis_jobs.
                output_path_str = str(audio_dir / f"{job['job_id']}.wav")
            output_path = Path(output_path_str)
            # Skip only when the checkpoint row already represents completed
            # audio.  Pending/failed rows may have refreshed text and must be
            # regenerated even if an old file exists at the same path.
            if (
                job.get("status") == "completed"
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
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
                retry_output = j.get("audio_path") or str(audio_dir / f"{j['job_id']}.wav")
                engine_map[j["engine"]].synthesize(
                    j["text"], j["voice"], retry_output
                )
                if Path(retry_output).exists():
                    db.mark_completed(j["job_id"], retry_output)

        # Free GPU memory after each XTTS engine
        if hasattr(engine, "unload"):
            engine.unload()

    # ── Build synthesis manifest with quality signals ─────────────────────────
    log.info("Running quality checks and building manifest …")
    # Use get_completed_jobs() so the manifest always reflects the text that was
    # *actually synthesized* (stored in the DB at job-creation time), not the
    # potentially different text from the current run's prompts.
    completed = db.get_completed_jobs()  # {job_id: full_job_dict}

    manifest_path = manifest_dir / f"manifest_{run_id}.jsonl"
    written = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for job in tqdm(jobs, desc="Quality checks"):
            job_id = job["job_id"]
            db_job = completed.get(job_id)
            if db_job is None:
                continue  # not completed — skip

            audio_path = db_job["audio_path"]
            qr = quality_check(audio_path, quality_cfg)

            record = {
                "id": job_id,
                "prompt_id": db_job["prompt_id"],
                "text": db_job["text"],
                "domain": db_job.get("domain") or "",
                "source": db_job.get("source") or "",
                "engine": db_job["engine"],
                "voice": db_job["voice"],
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

    # ── Apply noise augmentation to a random subset ────────────────────────────
    add_noise_ratio = syn_cfg.get("add_noise_ratio", 0.0)
    if add_noise_ratio > 0 and written > 0:
        log.info("Applying noise augmentation to %.0f%% of clips …", add_noise_ratio * 100)
        
        # Read manifest
        manifest_records = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    manifest_records.append(json.loads(line))
        
        # Select random indices for noise
        num_to_augment = max(1, int(len(manifest_records) * add_noise_ratio))
        indices_to_augment = random.sample(range(len(manifest_records)), num_to_augment)
        
        for idx in indices_to_augment:
            record = manifest_records[idx]
            audio_path = record.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                continue
            
            # Add noise with SNR ~20dB (moderate noise)
            success = add_gaussian_noise(audio_path, audio_path, snr_db=20.0)
            if success:
                record["has_noise_augmentation"] = True
                log.debug("Added noise to %s", audio_path)
            else:
                log.warning("Failed to add noise to %s", audio_path)
        
        # Rewrite manifest with augmentation flags
        with manifest_path.open("w", encoding="utf-8") as fh:
            for record in manifest_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        log.info("Noise augmentation complete — %d clips modified.", num_to_augment)
    
    return manifest_path
