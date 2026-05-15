# SSDP — Implementation Plan

## Overview

A 4-stage pipeline that produces a **synthetic Egyptian Arabic speech dataset** ready for STT fine-tuning.

```
[Stage 1: Prompt Generation]
        ↓
[Stage 2: TTS Synthesis]
        ↓
[Stage 3: Review UI]
        ↓
[Stage 4: Export]
```

---

## Technology Choices

| Concern | Choice | Reason |
|---|---|---|
| Prompt generation | **Gemini API** (`gemini-1.5-flash`) | Free tier, strong Arabic/dialect support |
| Seed corpus | Pre-written static sentences (Egyptian Arabic domains) | Immediate offline fallback, no API needed for first run |
| TTS engine 1 | **edge-tts** (`ar-EG-SalmaNeural`, `ar-EG-ShakirNeural`) | Free, no quota, good Egyptian Arabic voice quality |
| TTS engine 2 | **Coqui XTTS v2** (local) | Open-source, zero-cost, multilingual, good for diversity |
| TTS engine 3 | **Google Vertex AI TTS** | Highest quality; used selectively, quotas respected |
| Review UI | **Gradio** | Simple web UI, runs locally, audio widget built-in |
| Config management | **YAML + python-dotenv** | Externalized, version-controlled template |
| Output format | **HuggingFace `datasets`-compatible (Parquet + metadata JSONL)** | Standard STT fine-tuning format |
| Async/job handling | **Python `asyncio` + `tqdm`** with checkpointing to SQLite | Resume interrupted synthesis runs |

---

## Project Structure

```
TTS-pipeline/
├── config/
│   ├── config.yaml            # All tunable settings (batch size, engines, etc.)
│   └── .env.example           # API key template
├── data/
│   ├── seeds/
│   │   └── seed_sentences.txt # Static Egyptian Arabic seed corpus
│   ├── prompts/               # Generated .jsonl prompt files
│   ├── audio/                 # Synthesized .wav files
│   ├── manifests/             # Per-run manifest .jsonl (text + audio path + metadata)
│   └── export/                # Final training-ready dataset
├── pipeline/
│   ├── __init__.py
│   ├── stage1_generate.py     # Prompt generation (Gemini + seeds)
│   ├── stage2_synthesize.py   # TTS synthesis (edge-tts / XTTS / Vertex)
│   ├── stage3_review.py       # Gradio review app
│   └── stage4_export.py       # Dataset packaging & export
├── utils/
│   ├── checkpointing.py       # SQLite-based job state tracker
│   ├── quality.py             # Automated quality signals (SNR, duration checks)
│   └── arabic_utils.py        # Text normalization, diacritic handling, edge cases
├── tests/
│   ├── test_generate.py
│   ├── test_synthesize.py
│   └── test_export.py
├── run_pipeline.py            # End-to-end runner (CLI)
├── requirements.txt
├── README.md
└── plan.md                    ← this file
```

---

## Stage 1 — Prompt Generation

**Goal:** Produce a diverse, high-quality corpus of Egyptian Arabic text prompts.

**Approach:**
1. Start with a **static seed corpus** (~100 hand-written sentences) covering:
   - Daily conversation (greetings, shopping, transport)
   - Numbers, dates, times (common STT failure points)
   - Proper nouns (Egyptian cities, names)
   - Code-switching phrases (Egyptian Arabic often mixes with English/MSA)
   - Colloquial expressions and filler words (يعني، بقى، اهو)
2. Use **Gemini API** to expand the seed corpus by:
   - Paraphrasing existing sentences
   - Generating new sentences in specified domains
   - Varying sentence length (short 3-word utterances → long 20-word sentences)
3. Text normalization pass:
   - Strip diacritics optionally (configurable)
   - Normalize Arabic numerals
   - Handle common Egyptian spelling variants (e.g., ج vs. ق usage in Egyptian dialect)

**Output:** `data/prompts/prompts_<run_id>.jsonl`
```json
{"id": "p001", "text": "إيه اللي بتدور عليه؟", "domain": "conversation", "source": "seed"}
```

---

## Stage 2 — TTS Synthesis

**Goal:** Synthesize audio for every approved prompt, producing clean WAV files.

**Approach:**
- **Multi-engine support** with a unified `TTSEngine` interface
- Engine selection configurable per run (or randomly assigned for diversity)
- Engines:
  - `edge-tts` — async, batched, rate-limit safe
  - `Coqui XTTS v2` — local batch inference
  - `Google Vertex AI TTS` — cloud, used sparingly
- Each prompt gets one or more voices to maximize speaker diversity

**Resilience:**
- All synthesis tracked in **SQLite checkpoint DB** (`data/synthesis.db`)
- Failed items retried up to N times (configurable)
- Skips already-synthesized items on re-run (resumable)
- Batching with configurable `batch_size`

**Output:** `data/audio/<prompt_id>_<engine>_<voice>.wav` + updated manifest JSONL

---

## Stage 3 — Review UI (Gradio)

**Goal:** Allow a human reviewer to quickly approve or reject (text, audio) pairs.

**UI Features:**
- Shows one sample at a time: Arabic text + audio player
- Buttons: ✅ Approve / ❌ Reject / ⏭ Skip
- Optional free-text note field (e.g., "wrong pronunciation", "background noise")
- Progress bar showing reviewed/total
- Filter view: show only un-reviewed, rejected, or approved
- Saves decisions back to the manifest JSONL in real time

**Automated Pre-filtering (before human review):**
- SNR check — flag clips below threshold (too quiet or noisy)
- Duration check — flag clips that are suspiciously short or long
- Silence detection — flag clips with leading/trailing silence > threshold
- These signals are shown as badges in the UI but don't auto-reject

**Output:** Manifest updated with `"review_status": "approved" | "rejected" | "skipped"` and optional `"review_note"`

---

## Stage 4 — Export

**Goal:** Package approved samples into a training-ready dataset.

**Format:** HuggingFace `datasets`-compatible layout
```
data/export/
├── train/
│   ├── metadata.jsonl         # {"file_name": "audio/p001.wav", "transcription": "..."}
│   └── audio/
│       ├── p001.wav
│       └── ...
└── dataset_info.json          # Stats: total samples, duration, engines used, etc.
```

**Why this format:**
- Directly loadable with `datasets.load_dataset("audiofolder", ...)`
- Compatible with `transformers` Whisper fine-tuning scripts out of the box
- Simple, inspectable, no proprietary dependencies

**Also exports:**
- Summary stats (total duration, samples per engine, samples per domain)
- Rejected sample log (for audit / future analysis)

---

## End-to-End Runner

`run_pipeline.py` will expose a CLI:

```bash
python run_pipeline.py --stage all            # Run all 4 stages
python run_pipeline.py --stage generate       # Stage 1 only
python run_pipeline.py --stage synthesize     # Stage 2 only
python run_pipeline.py --stage review         # Launch Gradio UI
python run_pipeline.py --stage export         # Stage 4 only
python run_pipeline.py --resume               # Resume from last checkpoint
```

---

## Egyptian Arabic Challenges — Addressed

| Challenge | How the Pipeline Addresses It |
|---|---|
| Dialect inconsistency (EGY vs. MSA) | Seed corpus hand-written in EGY; Gemini prompted explicitly for dialect |
| Spelling variation (ق→أ, ث→س, etc.) | `arabic_utils.py` normalization + configurable variant expansion |
| Code-switching | Seed corpus includes mixed phrases; handled in text normalization |
| TTS mispronunciation of dialect words | Multi-engine diversity + human review stage to catch bad pronunciations |
| Diacritics | Configurable strip/keep; majority of EGY text is undiacritized |
| Short/clipped audio | Duration + silence checks in automated quality signals |

---

## Optional Advanced Items (Planned)

- [ ] Caching layer for Gemini API calls (avoid re-generating same domain)
- [ ] Unit tests for `arabic_utils`, checkpointing, export format
- [ ] Automated quality score stored in manifest (SNR value, not just flag)
- [ ] README with architecture diagram, trade-offs, and sample dataset

---

## Milestones

1. **Scaffold** — project structure, config, requirements
2. **Stage 1** — seed corpus + Gemini generation
3. **Stage 2** — edge-tts synthesis with checkpointing
4. **Stage 2+** — XTTS + Vertex engine adapters
5. **Stage 3** — Gradio review UI
6. **Stage 4** — export to HF-compatible format
7. **Quality signals** — SNR, duration, silence detection
8. **Tests + README**
