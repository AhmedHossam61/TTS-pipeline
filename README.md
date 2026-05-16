# Synthetic Speech Data Pipeline (S.S.D.P.)

A four-stage pipeline for generating, synthesizing, reviewing, and exporting a synthetic Egyptian Arabic speech dataset suitable for fine-tuning STT models.

Built as an Olimi AI case study.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure secrets (Gemini / Vertex API keys)
cp config/.env.example config/.env

# Run the full pipeline (generates 10 prompts by default)
python run_pipeline.py --stage all

# Resume an interrupted run
python run_pipeline.py --stage all --run-id run_20260516_120526

# Run individual stages
python run_pipeline.py --stage generate
python run_pipeline.py --stage synthesize
python run_pipeline.py --stage review      # opens Gradio UI at http://127.0.0.1:7860
python run_pipeline.py --stage export

# Evaluate synthesized audio against a pre-trained STT model (WER / CER)
python auto_model_evals.py \
  --manifest data/manifests/manifest_<run_id>.jsonl \
  --backend vertex \
  --language-code ar-EG
```

Key config knobs are in `config/config.yaml` — change `num_prompts`, enable/disable engines, adjust quality thresholds, or flip `expand_with_gemini: true` to use Gemini for richer prompt generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         run_pipeline.py                             │
│            (CLI — stage selection, run ID, config override)         │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │  Stage 1 — Prompt Generation                     │
        │  stage1_generate.py                              │
        │                                                  │
        │  Seed sentences (175) + optional Gemini expand   │
        │  → text normalization (diacritics, dedup)        │
        │  → data/prompts/prompts_<run_id>.jsonl           │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │  Stage 2 — TTS Synthesis                         │
        │  stage2_synthesize.py                            │
        │                                                  │
        │  Per prompt × engine × voice job                 │
        │  ┌────────────┐ ┌────────┐ ┌────────────────┐   │
        │  │  edge-tts  │ │ egtts  │ │  gemini-tts    │   │
        │  │  (free,    │ │(XTTS-2 │ │  (Vertex AI,   │   │
        │  │  no GPU)   │ │fine-   │ │  best quality) │   │
        │  │            │ │tune)   │ │                │   │
        │  └────────────┘ └────────┘ └────────────────┘   │
        │  → quality check (SNR, duration, silence)        │
        │  → data/audio/*.wav + manifest JSONL             │
        │  → SQLite checkpoint DB (resumable)              │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │  Stage 3 — Human Review                          │
        │  stage3_review.py (Gradio UI, port 7860)         │
        │                                                  │
        │  Text + audio per sample, quality badges         │
        │  A = approve / R = reject / N = next             │
        │  → review_status written back to manifest        │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │  Stage 4 — Export                                │
        │  stage4_export.py                               │
        │                                                  │
        │  approved samples only                           │
        │  → HuggingFace audiofolder format               │
        │  → data/export/train/ + data/export/test/       │
        │  → 90 / 10 train-test split                     │
        └──────────────────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │  Evaluation (optional)                           │
        │  auto_model_evals.py                            │
        │                                                  │
        │  WER / CER per engine and domain                 │
        │  Backends: Vertex STT or predictions JSONL       │
        └──────────────────────────────────────────────────┘
```

Every stage is independently runnable. A `run_id` (e.g. `run_20260516_120526`) threads through all four stages, linking prompts → audio → review decisions → export.

---

## Engine Choices

| Engine | Model | GPU | Cost | Dialect authenticity |
|--------|-------|-----|------|----------------------|
| `edge-tts` | ar-EG-SalmaNeural / ar-EG-ShakirNeural | No | Free | ⚠ MSA prosody (see below) |
| `egtts` | OmarSamir/EGTTS-V0.1 (XTTS-v2 fine-tune) | Yes (CUDA) | Free (local) | ✅ Egyptian Arabic |
| `gemini-tts` | gemini-3.1-flash-tts-preview via Vertex AI | No | Pay-per-use | ✅ Good Egyptian Arabic |

**Why three engines?** Acoustic diversity. A dataset generated from a single TTS voice produces a biased STT model — the model learns the voice's artifacts rather than natural speech. Three different synthesis stacks (neural streaming, XTTS voice cloning, LLM-based TTS) give the downstream model a wider acoustic space to generalise from.

**Why edge-tts?** Zero infrastructure, no GPU, no API key. It handles batch generation without rate limits and is useful as a fast, free baseline. Its dialect quality is lower (see Known Quality Issues), but the audio itself is clean.

**Why egtts?** The most authentic Egyptian Arabic voice in the set. XTTS-v2 fine-tuned specifically on Egyptian Arabic colloquial speech. Requires a reference WAV for voice cloning and a CUDA-capable GPU.

**Why gemini-tts?** Highest overall audio quality and naturalness. Requires a Google Cloud project with Vertex AI enabled and an `ar-EG` language code. Use Charon or Achernar voices (configured by default).

Voice weight sampling (configured per engine in `config.yaml`) ensures that not every prompt goes to the same voice within an engine, further reducing redundancy.

---

## Review Approach

Stage 3 opens a Gradio web UI at `http://127.0.0.1:7860`. For each sample you see:

- The Arabic source text (RTL, large font)
- An inline audio player
- Metadata badges: domain, engine, voice, duration, SNR dB, quality flags
- A free-text notes field

**Keyboard shortcuts:** `A` = approve, `R` = reject, `N` = next, with a jump-to-pending button for large queues.

Decisions are written back to the manifest immediately — no save button, no data loss on browser refresh or crash. Only `approved` samples pass through to Stage 4 export.

Automated quality signals (SNR, duration, silence ratio) are surfaced as flags but do not auto-reject — they highlight candidates for rejection, leaving final judgment to the human reviewer.

---

## Output Format

Stage 4 produces a [HuggingFace `audiofolder`](https://huggingface.co/docs/datasets/audio_dataset) dataset:

```
data/export/
├── train/
│   ├── metadata.jsonl      # one record per approved sample
│   └── audio/              # WAV files at 24 kHz
├── test/
│   ├── metadata.jsonl
│   └── audio/
├── dataset_info.json       # run summary, counts, duration, engines, domains
└── rejected_log.jsonl      # audit trail of excluded samples
```

Load directly with:

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="data/export")
```

Each `metadata.jsonl` record:

```json
{
  "id":             "run_xxx_p0001_edge-tts_ar_EG_SalmaNeural",
  "text":           "إزيك النهارده؟",
  "domain":         "conversation",
  "engine":         "edge-tts",
  "voice":          "ar-EG-SalmaNeural",
  "audio_path":     "audio/run_xxx_p0001_edge-tts_ar_EG_SalmaNeural.wav",
  "sample_rate":    24000,
  "duration_sec":   1.23,
  "snr_db":         28.5,
  "quality_flags":  [],
  "review_status":  "approved"
}
```

Default train/test split: **90% / 10%**, deterministic (fixed seed).

---

## Known Quality Issues

### edge-tts — Modern Standard Arabic (MSA) prosody

`ar-EG-SalmaNeural` and `ar-EG-ShakirNeural` are Egyptian Arabic voices in name, but their prosody and intonation patterns are closer to Modern Standard Arabic (فصحى) than to natural Egyptian colloquial speech (عامية مصرية). Word stress, sentence rhythm, and vowel reduction do not match how Egyptians speak in everyday conversation.

**Impact on downstream STT:** A model trained predominantly on edge-tts audio may generalise poorly to real Egyptian speech. Mitigations in this pipeline: edge-tts clips share the dataset with egtts and gemini-tts clips, which have more authentic dialect character. Treat edge-tts audio as a clean-acoustic supplement, not a dialect ground truth.

### egtts — single speaker identity

All egtts clips share one reference audio file (`data/references/egtts_reference.wav`). XTTS voice cloning reproduces the timbre of that reference speaker across every sample. There is no speaker diversity within the egtts engine's contribution to the dataset.

### Synthetic data general risks

All three engines produce unnaturally clean audio: no background noise, no filler words (آه / إيه / يعني), no disfluencies, no natural pauses or breath sounds. The pipeline applies Gaussian noise augmentation to 40% of clips (`add_noise_ratio: 0.4`) to partially simulate real-world recording conditions, but this is not a substitute for natural speech variation.

**Recommendation:** Mix this dataset with real Egyptian Arabic speech recordings before fine-tuning an STT model.

---

## Configuration Reference

All settings live in `config/config.yaml`. Secrets (API keys) go in `config/.env`.

```yaml
pipeline:
  run_id: null              # auto-generated if null
  data_dir: data

generation:
  num_prompts: 10           # total prompts to generate
  expand_with_gemini: false # set true to use Gemini for additional prompts
  strip_diacritics: true

synthesis:
  batch_size: 10
  max_retries: 3
  sample_rate: 24000
  add_noise_ratio: 0.4      # fraction of clips that get Gaussian noise added

  quality:
    min_duration_sec: 0.5
    max_duration_sec: 30.0
    min_snr_db: 10.0
    max_silence_ratio: 0.35

  engines:
    edge-tts:
      enabled: true
      voices:
        - name: ar-EG-SalmaNeural
          weight: 0.5
        - name: ar-EG-ShakirNeural
          weight: 0.5

    egtts:
      enabled: true
      model_repo: OmarSamir/EGTTS-V0.1
      reference_wav: data/references/egtts_reference.wav

    gemini-tts:
      enabled: true
      model: gemini-3.1-flash-tts-preview
      voices:
        - name: Charon
          weight: 0.5
        - name: Achernar
          weight: 0.5

export:
  train_ratio: 0.9
  filter_status: approved
```

---

## Project Structure

```
TTS-pipeline/
├── pipeline/
│   ├── stage1_generate.py      # Prompt generation
│   ├── stage2_synthesize.py    # TTS synthesis (edge-tts, egtts, gemini-tts)
│   ├── stage3_review.py        # Gradio review UI
│   └── stage4_export.py        # HuggingFace audiofolder export
├── utils/
│   ├── arabic_utils.py         # Diacritics removal, text normalization
│   ├── quality.py              # SNR, duration, silence ratio checks
│   ├── audio_augmentation.py   # Gaussian noise augmentation
│   └── checkpointing.py        # SQLite job state machine
├── auto_model_evals.py         # WER / CER evaluation against STT backends
├── run_pipeline.py             # Main CLI entry point
├── config/config.yaml          # All pipeline configuration
├── data/
│   ├── seeds/seed_sentences.txt    # 175 curated Egyptian Arabic sentences
│   ├── references/                 # Reference WAVs for voice cloning
│   ├── audio/                      # Synthesized WAV files
│   ├── manifests/                  # Per-run JSONL manifests
│   ├── export/                     # Final train/test dataset
│   └── synthesis.db                # SQLite checkpoint database
└── tests/                          # pytest suite
```

---

## Trade-offs

**Quantity vs. authenticity:** The seed bank (175 sentences) produces a small but curated corpus. Enabling `expand_with_gemini` can scale to 1,000+ prompts cheaply, but Gemini-generated sentences are less idiomatic than hand-curated ones. The default (seeds only) favours quality over volume.

**Speed vs. GPU dependency:** edge-tts is fast and requires no local hardware. egtts requires a CUDA GPU and takes longer per clip. If GPU availability is a constraint, disable egtts in config and rely on edge-tts + gemini-tts.

**Free vs. metered:** edge-tts and egtts are free. gemini-tts bills per character via Vertex AI. For large runs, monitor GCP spend or cap `num_prompts` accordingly.

**Automated vs. human quality gates:** Quality checks (SNR, duration, silence) flag bad clips but do not auto-reject them. This keeps reviewer control over borderline cases but increases review burden for large datasets. Hard failures (duration < 0.3s or SNR < 5 dB) are good candidates for auto-rejection if reviewer time is scarce.

**Single reference speaker for egtts:** Using one reference WAV per engine is simple and reproducible, but all egtts clips share the same speaker timbre. For speaker diversity, provide multiple reference WAVs and cycle through them per prompt.
