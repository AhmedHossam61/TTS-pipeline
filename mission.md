# Mission: Synthetic Speech Data Pipeline (S.S.D.P.)
> Olimi AI Case Study — 72-hour timebox
>
> **Goal:** Build a pipeline that produces a synthetic Egyptian Arabic speech dataset suitable for fine-tuning an STT model.

---

## Core Expectations

### 1. Prompt Generation ✅
Produce a corpus of Egyptian Arabic text prompts to feed into TTS.

- [x] Seed sentence bank loaded from `data/seeds/seed_sentences.txt` (175 sentences)
- [x] Domain-aware sampling across 14 domains (conversation, shopping, health, etc.)
- [x] Optional Gemini API expansion (configurable via `expand_with_gemini`)
- [x] Text normalization — diacritics removal, deduplication (`utils/arabic_utils.py`)
- [x] JSONL output with `id`, `text`, `domain`, `source`, `char_count`, `created_at`
- [x] Rationale documented in `PROJECT_CONSTITUTION.md`

> **Suggestion:** The seed bank (175 sentences) is a solid start, but for a richer STT training set you'd want 1,000+ diverse prompts. Enabling Gemini expansion (`expand_with_gemini: true`) and using structured domain prompt templates would scale this cheaply. Also consider adding code-switched sentences (Arabic + English words inline), which are very common in real Egyptian speech but likely underrepresented in seeds.

---

### 2. TTS Synthesis ✅
Synthesize audio from the prompts.

**Active engines (3):**
| Engine | Model | Quality Note |
|---|---|---|
| `edge-tts` | ar-EG-SalmaNeural / ar-EG-ShakirNeural | Clean audio, MSA prosody (documented in README) |
| `egtts` | OmarSamir/EGTTS-V0.1 (XTTS-v2 fine-tune) | Authentic Egyptian Arabic voice — fully working |
| `gemini-tts` | `gemini-3.1-flash-tts-preview` via Vertex AI | Best overall audio quality |

- [x] All 3 engines working and producing audio (`data/audio/`) — 30 files across 10 prompts
- [x] **Voice weight sampling** — each prompt gets one voice per engine drawn by `voice_weights` (e.g., 50/50 Salma/Shakir for edge-tts, 50/50 Charon/Achernar for gemini-tts), reducing redundancy and generation time
- [x] TorchCodec / FFmpeg bug resolved — `egtts` generating successfully and reliably
- [x] `torchaudio.load/save` monkey-patched to `soundfile` at module import time (before TTS imports) to bypass Windows DLL dependency
- [x] Engine-agnostic architecture — new engines plug in via config
- [x] Unified 24 kHz WAV output
- [x] SQLite checkpoint DB (`data/synthesis.db`) — job state machine: `pending → running → completed/failed`
- [x] Crash recovery via `reset_running()` on startup
- [x] Configurable batch size (`10`) and max retries (`3`)
- [x] Windows FFmpeg DLL path auto-injection
- [x] Rationale documented in `README.md` (Engine Choices section) and `PROJECT_CONSTITUTION.md`

> **Suggestion:** `gemini-3.1-flash-tts-preview` is the strongest engine for quality. If Vertex AI availability is a concern, consider making it primary with `egtts` as fallback rather than treating all three as equal weights. The known `edge-tts` MSA prosody limitation is now documented in `README.md` (Known Quality Issues section).

---

### 3. Review ✅
Provide a way to review (text, audio) pairs and separate good samples from bad before training.

**Current state:** 30 samples reviewed — 20 approved / 10 rejected.

- [x] Gradio web UI (`pipeline/stage3_review.py`, default: `http://127.0.0.1:7860`)
- [x] Arabic RTL text display with large font
- [x] Inline audio player per sample
- [x] Metadata badges: domain, engine, voice, duration, SNR, quality flags
- [x] Review states: `approved`, `rejected`, `pending`, `skipped`
- [x] Free-text review notes field
- [x] Previous / Next navigation + jump-to-pending
- [x] Real-time persistence — decisions survive browser refresh and app restart
- [x] Summary stats (approved / rejected / pending counts)
- [x] **Keyboard shortcuts** — `A` = approve, `R` = reject, `N` = next sample (implemented via `keydown` listener in Gradio JS)

> **Suggestion:** Consider a one-click "auto-reject all `quality_passed = false`" button as a first-pass filter, leaving human review for borderline cases only. This would speed up review of large datasets (500+ clips) significantly.

---

### 4. Training-Ready Output ✅
Export the reviewed dataset in a structured form ready for downstream training.

- [x] HuggingFace `audiofolder` format (compatible with `datasets.load_dataset("audiofolder", ...)`)
- [x] Train / test split (default 90/10, deterministic for reproducibility)
- [x] Output: `data/export/train/` and `data/export/test/` each with `metadata.jsonl` + `audio/`
- [x] `dataset_info.json` — run metadata, sample counts, total duration, engines, domains
- [x] `rejected_log.jsonl` — audit trail of excluded samples
- [x] Filters on `review_status = approved` (configurable)
- [x] Format rationale documented in `README.md` (Output Format section)

> **Suggestion:** Add a `dataset_card.md` (HuggingFace model card format) to the export with language tag (`ar-EG`), data source, license, and engine breakdown. This would make the dataset directly publishable to HuggingFace Hub with no extra steps.

---

### Pipeline Service ✅
- [x] Runnable end-to-end via `python run_pipeline.py`
- [x] Stage selection: `--stage [all|generate|synthesize|review|export]`
- [x] Resume existing run: `--run-id <id>`
- [x] Long-running synthesis handled via async batching + SQLite checkpointing
- [x] Configuration fully externalized in `config/config.yaml`
- [x] Secrets in `config/.env` (not committed)
- [x] Intermediate artifacts: per-run JSONL manifests, timestamped logs, SQLite DB

---

## Advanced (Optional, High-Impact)

### Awareness of Synthetic Data Bias ✅
How synthetic data can mislead a downstream STT model, and how the pipeline addresses it.

- [x] 3 distinct TTS engines reduce single-voice/single-engine acoustic bias
- [x] Voice weight sampling introduces speaker variation within each engine (2 voices for edge-tts, 2 for gemini-tts)
- [x] Gaussian noise augmentation applied to 40% of clips (`add_noise_ratio: 0.4`) to simulate real-world recording conditions
- [x] **Written documentation** in `README.md` (Known Quality Issues + Trade-offs sections) covering: TTS prosody artifacts, MSA vs. dialect gap, unnaturally clean recordings, noise augmentation rationale, recommendation to mix with real data before fine-tuning
- [ ] **Missing:** All `egtts` clips share one speaker identity (single reference audio) — no speaker diversity for that engine

> **Suggestion:** For a production dataset, cycle through multiple reference WAVs for `egtts` to introduce speaker variation. Also add a "Synthetic Data Risks" subsection to `README.md` that explicitly calls out filler words (آه / إيه / يعني), disfluencies, and over-representation of short sentences as gaps — these are explicit evaluation criteria at Olimi AI.

---

### Automated Quality Signals ✅
- [x] SNR estimation (energy percentile method, `utils/quality.py`)
- [x] Duration validation (min 0.5s / max 30s, configurable)
- [x] Silence ratio detection (max 35%, configurable)
- [x] Descriptive quality flags written to manifest (`too_short`, `low_snr`, etc.)
- [x] `quality_passed` boolean surfaced in review UI with red badge highlighting
- [x] Quality thresholds configurable in `config/config.yaml`

> **Suggestion:** Currently `quality_passed = false` is a warning, not a gate — clips still enter the review queue. Consider an auto-reject path for hard failures (duration < 0.3s or SNR < 5 dB) to reduce reviewer burden. Soft failures stay in the queue for human judgement.

---

### Edge Case Handling in Source Text ⬜ Partial
- [x] Diacritics removal (`strip_diacritics: true` in config)
- [x] Prompt deduplication before synthesis
- [x] Arabic text validation in `utils/arabic_utils.py`
- [ ] **Missing:** No numeral normalization — sentences with "٣ كيلو" or "2024" will be read inconsistently across engines
- [ ] **Missing:** No handling of code-switched text (Arabic + Latin characters, common in Egyptian informal writing)
- [ ] **Missing:** No maximum character length guard before sending to TTS (very long prompts can cause XTTS failures)

> **Suggestion:** Add a pre-synthesis normalization step: expand Arabic/Western numerals to words, strip or transliterate stray Latin characters, and cap prompts at ~200 characters. This belongs in Stage 1 output or as a pre-processing hook in Stage 2 before each synthesis call.

---

### Caching, Retries, Resumability ✅
- [x] SQLite checkpoint DB persists all job state across runs
- [x] Completed jobs skipped on re-run (idempotent)
- [x] Failed jobs retried up to `max_retries: 3`
- [x] `reset_running()` recovers jobs stuck in `running` state after crash
- [x] HuggingFace model cache used by `snapshot_download` (no re-download on restart)
- [x] Audio file existence check as extra safety net before re-synthesis

---

### Tests for Critical Logic ⬜ Partial
- [x] `tests/test_generate.py` — seed loading, deduplication, Gemini mock, manifest writing
- [x] `tests/test_export.py` — approved/rejected filtering, metadata fields, split generation
- [x] `tests/test_arabic_utils.py` — diacritics removal, normalization, Arabic validation
- [ ] **Missing:** No tests for `stage2_synthesize.py` (checkpoint DB, engine dispatch, voice weight sampling, retry logic)
- [ ] **Missing:** No tests for `stage3_review.py` (manifest persistence, state transitions)
- [ ] **Missing:** No tests for `utils/quality.py` (SNR/duration/silence logic)
- [ ] **Missing:** No integration test for the full `run_pipeline.py` end-to-end flow

> **Suggestion:** The checkpoint DB (`utils/checkpointing.py`) is the most critical piece for reliability and has no test coverage. A test that simulates a mid-run crash (mark some jobs `running`, call `reset_running()`, verify they return to `pending`) would directly validate the resumability guarantee.

---

### STT Evaluation (Beyond Task Scope) ✅
Automatic WER/CER evaluation against a pre-trained STT model — exceeds the case study requirements.

- [x] `auto_model_evals.py` — evaluates synthesized audio manifest against Vertex STT or precomputed predictions JSONL
- [x] WER and CER computed per sample, with breakdowns by engine and domain
- [x] Predictions JSONL schema enables future fine-tuned model evaluation without changing the script
- [x] Arabic text normalization applied before metric computation (`utils/arabic_utils.py`)

---

## Deliverables

- [x] **Source code** — fully implemented, 4-stage pipeline + evaluation script
- [x] **README** ✅ — `README.md` written covering: quick-start, architecture diagram, engine choices rationale, review approach, output format, known quality issues (edge-tts MSA prosody, egtts single speaker, synthetic data risks), trade-offs, and configuration reference
- [x] **Sample dataset** ✅ — 30 synthesized audio files (`data/audio/`), 20 approved / 10 rejected after Stage 3 review. Run `python run_pipeline.py --stage export` to produce the final `data/export/train/` and `data/export/test/` HuggingFace-format output

> **Remaining optional steps:**
> 1. Run `python run_pipeline.py --stage export` to materialise `data/export/` from the 20 approved samples
> 2. Add a `dataset_card.md` to the export for HuggingFace Hub compatibility
> 3. Add test coverage for `stage2_synthesize.py` and `utils/quality.py`
> 4. Add numeral normalization and character length guard to pre-synthesis text handling
