# SSDP Session Notes

## Environment and Tooling
- OS: Windows.
- Project path: G:/AI_ML/Projects/TTS-pipeline.
- Python: 3.13 in virtual environment.
- GPU: NVIDIA RTX 3050 8GB.
- Driver reports CUDA runtime 13.2 (from nvidia-smi).

## Dependency Lessons
- Original TTS package pin (TTS==0.22.0) fails on Python 3.13 due to outdated build tooling path.
- Using coqui-tts works on Python 3.13 and keeps the same TTS import namespace.
- For PyTorch on this machine, keep torch and torchaudio on matching CUDA wheel index and matching versions.
- cu121 install path failed for this Python version.
- cu132 did not provide a usable torchaudio wheel path in this setup.
- Working path used here: install both torch and torchaudio from cu128 index with pip.
- Use pip (not uv pip) for PyTorch custom index installs to avoid resolver/index issues.

## Pipeline Behavior Notes
- Stage 1 (generate) is still useful even when seeds exist.
- Stage 1 packages prompts into JSONL schema, deduplicates, normalizes text, and can expand with Gemini.
- Gemini is used only in Stage 1.
- Stages 2 to 4 do not call Gemini.
- domain metadata is propagated to later stages and export metadata.
- source metadata helps distinguish seed versus Gemini generated prompts.

## .env and Auth Notes
- Stage 1 currently reads GEMINI_API_KEY directly.
- Vertex-related variables in .env are not consumed by current Stage 1 implementation.
- Keep secrets in config/.env only and never commit real keys.

## Reference Audio Notes
- Attempt to load KickItLikeShika/NileTTS dataset returned 401 Unauthorized in this session.
- Result: dataset is private, gated, missing, or inaccessible with current auth.
- Reference WAVs were successfully produced via edge-tts plus conversion workflow:
  - data/references/egtts_reference.wav
  - data/references/niletts_reference.wav

## Practical Run Order Used
1. Install dependencies (including matching torch + torchaudio).
2. Ensure config/.env contains GEMINI_API_KEY.
3. Generate prompts with Stage 1.
4. Ensure reference WAVs exist.
5. Continue with synthesis, review, and export stages.

## Common Failure Patterns and Fixes
- Error: No matching distribution for torch/torchaudio on a CUDA index.
  - Fix: switch to an index that publishes wheels for your Python ABI and install both packages together.
- Error: torchaudio library load failure on import.
  - Fix: reinstall torch and torchaudio from the same index/version family.
- Error: GEMINI_API_KEY not set, Gemini expansion skipped.
  - Fix: set GEMINI_API_KEY in config/.env.
- Error: 401 loading NileTTS dataset.
  - Fix: use local reference WAVs fallback.

## Current State Snapshot
- generate stage ran successfully.
- reference WAVs are present.
- project is ready to proceed with synthesize stage.
