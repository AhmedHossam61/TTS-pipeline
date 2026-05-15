# Troubleshooting Log

## 1. TTS package failed on Python 3.13
- Symptom: `TTS==0.22.0` failed to build/install on Windows with Python 3.13.
- Cause: the old PyPI TTS package relies on outdated build tooling and setuptools behavior.
- Fix used: install `coqui-tts` instead. It keeps the same `TTS` import namespace.

## 2. PyTorch wheel index mismatch
- Symptom: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` failed with no matching distribution.
- Cause: that index did not provide compatible wheels for this Python 3.13 setup.
- Fix used: install `torch` and `torchaudio` from the `cu128` index and keep both on the same CUDA family.

## 3. torchaudio import/load failure
- Symptom: `torchaudio` failed to load native libraries after mixed installs.
- Cause: mismatched versions/CUDA families between `torch`, `torchaudio`, and `torchvision`.
- Fix used:
  - reinstall `torch` and `torchaudio` from the same index family
  - remove incompatible `torchvision`

## 4. XTTS / Coqui imports failed
- Symptom: importing `from TTS.tts.models.xtts import Xtts` failed.
- Causes found:
  - incompatible `transformers` version
  - incompatible leftover `torchvision`
  - missing `torchcodec`
- Fix used:
  - uninstall incompatible `torchvision`
  - install `transformers==4.57.1`
  - install `torchcodec`
- Verified result: `torch`, `torchaudio`, CUDA, and `TTS` imports all worked after alignment.

## 5. Reference dataset download failed with 401
- Symptom: loading `KickItLikeShika/NileTTS` dataset returned `401 Unauthorized`.
- Cause: dataset was private, gated, removed, or inaccessible in current environment.
- Fix used: generate local reference WAVs using `edge-tts` and convert them to WAV.
- Produced files:
  - `data/references/egtts_reference.wav`
  - `data/references/niletts_reference.wav`

## 6. Stage 2 crashed with `audio_path=None`
- Symptom: synthesis crashed with:
  - `TypeError: argument should be a str or an os.PathLike object ... not 'NoneType'`
- Cause: checkpoint DB rows were inserted without persisting `audio_path`.
- Fix used:
  - update checkpoint inserts to store `audio_path`
  - add runtime fallback to rebuild path from `job_id` for older DB rows
- Files updated:
  - `utils/checkpointing.py`
  - `pipeline/stage2_synthesize.py`

## 7. Repeated `Could not load libtorchcodec` during edge-tts synthesis
- Symptom: Stage 2 repeatedly logged TorchCodec / FFmpeg shared library errors during `edge-tts` synthesis.
- Root cause:
  - the `edge-tts` path generated MP3 output
  - MP3 was converted with `torchaudio.load(...)`
  - `torchaudio` used TorchCodec for decoding
  - TorchCodec required FFmpeg shared DLLs visible on PATH
- Additional environment issue:
  - the existing FFmpeg install was a non-shared/static build from Winget
  - the shell was not using the new shared build path
- Fix used:
  - install `Gyan.FFmpeg.Shared`
  - update Stage 2 to prepend shared FFmpeg `bin` on Windows automatically
  - replace edge MP3-to-WAV conversion to use `ffmpeg` directly instead of `torchaudio.load`
- File updated:
  - `pipeline/stage2_synthesize.py`

## 8. Gemini / Stage 1 clarification
- Observation: Gemini is not responsible for creating JSONL itself.
- Actual behavior:
  - Stage 1 code creates JSONL from seeds
  - Gemini only adds extra generated sentences when enabled
- Meaning:
  - Stage 1 still works without Gemini and still produces prompts JSONL

## 9. GPU usage clarification
- Observation: no GPU activity during Stage 1 generation is expected.
- Reason:
  - Stage 1 uses local text processing plus Gemini API on remote servers
  - local GPU is mainly relevant for XTTS synthesis in Stage 2
  - `edge-tts` does not use the local GPU

## 10. Current known-good stack
- Python: 3.13
- Torch: `2.11.0+cu128`
- Torchaudio: `2.11.0+cu128`
- Transformers: `4.57.1`
- Coqui TTS: `0.27.5`
- TorchCodec: installed
- FFmpeg: shared build required on Windows
