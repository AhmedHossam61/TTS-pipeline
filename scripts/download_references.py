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


if __name__ == "__main__":
    ok = _download_from_hf_dataset()
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
