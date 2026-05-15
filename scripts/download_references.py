"""
scripts/download_references.py
───────────────────────────────
Downloads sample reference audio files for the XTTS-based engines
(EGTTS-V0.1 and NileTTS) from HuggingFace Hub.

Reference audios are short (6–12 s) clean Egyptian Arabic speech clips
used for zero-shot voice conditioning in Coqui XTTS.

Usage:
    python scripts/download_references.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REFERENCES_DIR = Path("data/references")

# HuggingFace dataset that ships sample audio from the NileTTS project
# Dataset: KickItLikeShika/NileTTS  (Apache 2.0)
NILETTS_DATASET = "KickItLikeShika/NileTTS"

# We download one sample file per speaker and rename it for each engine.
# Adjust the filenames below if the dataset structure changes.
_SAMPLES = [
    # (dataset_repo,         subfolder,  filename_in_repo,  local_name)
    (NILETTS_DATASET, "data",  "train-00000-of-00001.parquet", None),
]


def _download_from_hf_dataset() -> bool:
    """
    Extract a short audio sample from the NileTTS HuggingFace dataset
    and save it as both reference files.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
        import soundfile as sf             # noqa: PLC0415
        import numpy as np                 # noqa: PLC0415
    except ImportError:
        log.error("Required packages missing.  Run: pip install datasets soundfile numpy")
        return False

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    egtts_path  = REFERENCES_DIR / "egtts_reference.wav"
    niletts_path = REFERENCES_DIR / "niletts_reference.wav"

    if egtts_path.exists() and niletts_path.exists():
        log.info("Reference files already exist — nothing to download.")
        return True

    log.info("Loading NileTTS dataset (this may take a moment on first run)…")
    try:
        ds = load_dataset(NILETTS_DATASET, split="train", streaming=True)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        return False

    saved = 0
    for sample in ds:
        audio = sample.get("audio")
        if audio is None:
            continue
        array = np.array(audio["array"], dtype=np.float32)
        sr    = int(audio["sampling_rate"])
        # Use a 6–10 second clip
        max_frames = sr * 10
        clip = array[:max_frames]

        if not egtts_path.exists():
            sf.write(str(egtts_path), clip, sr)
            log.info("Saved: %s", egtts_path)
            saved += 1

        if not niletts_path.exists():
            sf.write(str(niletts_path), clip, sr)
            log.info("Saved: %s", niletts_path)
            saved += 1

        if egtts_path.exists() and niletts_path.exists():
            break

    if saved:
        log.info(
            "\n✅  Reference files saved to %s\n"
            "   Update config/config.yaml if you want to use your own voice.",
            REFERENCES_DIR,
        )
        return True
    else:
        log.warning("No audio samples found in dataset.")
        return False


def _generate_via_edge_tts() -> bool:
    """
    Fallback: synthesise a short Egyptian Arabic sentence with edge-tts
    (free, no auth) and save it as both reference WAVs.
    """
    try:
        import asyncio
        import edge_tts  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import io
    except ImportError:
        log.error("Required packages missing.  Run: pip install edge-tts soundfile numpy")
        return False

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    egtts_path   = REFERENCES_DIR / "egtts_reference.wav"
    niletts_path = REFERENCES_DIR / "niletts_reference.wav"

    if egtts_path.exists() and niletts_path.exists():
        log.info("Reference files already exist — nothing to do.")
        return True

    # A ~8 second Egyptian Arabic reference sentence
    TEXT = (
        "أهلاً وسهلاً، ده نموذج صوتي بيتكلم بالعامية المصرية، "
        "وبيستخدم في تدريب نماذج تحويل النص للكلام."
    )

    async def _synth(voice: str, out_path: Path) -> None:
        if out_path.exists():
            return
        tmp_mp3 = out_path.with_suffix(".mp3")
        communicate = edge_tts.Communicate(TEXT, voice)
        await communicate.save(str(tmp_mp3))
        # Convert mp3 → wav via soundfile (requires pydub or ffmpeg fallback)
        try:
            import pydub  # noqa: PLC0415
            audio = pydub.AudioSegment.from_mp3(str(tmp_mp3))
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(str(out_path), format="wav")
        except ImportError:
            # pydub not available — try torchaudio
            import torchaudio  # noqa: PLC0415
            waveform, sr = torchaudio.load(str(tmp_mp3))
            waveform = torchaudio.functional.resample(waveform, sr, 22050).mean(0, keepdim=True)
            torchaudio.save(str(out_path), waveform, 22050)
        finally:
            if tmp_mp3.exists():
                tmp_mp3.unlink()
        log.info("Saved: %s", out_path)

    async def _run() -> None:
        await _synth("ar-EG-ShakirNeural", egtts_path)
        await _synth("ar-EG-SalmaNeural",  niletts_path)

    asyncio.run(_run())
    return egtts_path.exists() and niletts_path.exists()


if __name__ == "__main__":
    ok = _download_from_hf_dataset()
    if not ok:
        log.info("Trying edge-tts fallback to generate reference audio…")
        ok = _generate_via_edge_tts()
    if not ok:
        log.warning(
            "\nFallback: record your own 6–12 second reference WAV and place it at:\n"
            "  data/references/egtts_reference.wav\n"
            "  data/references/niletts_reference.wav\n"
            "\nffmpeg command (if you have a microphone):\n"
            "  ffmpeg -f dshow -i audio=\"Microphone\" -t 10 -ar 22050 -ac 1 "
            "data/references/egtts_reference.wav"
        )
        sys.exit(1)
