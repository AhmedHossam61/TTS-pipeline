"""
pipeline/stage4_export.py
─────────────────────────
Stage 4 — Training-Ready Export.

Packages approved samples into a HuggingFace ``audiofolder`` layout:

    data/export/
    ├── train/
    │   ├── metadata.jsonl          {"file_name": "audio/p0001_…wav", "transcription": "…"}
    │   └── audio/
    │       ├── p0001_edgetts_SalmaNeural.wav
    │       └── …
    ├── test/
    │   ├── metadata.jsonl
    │   └── audio/
    └── dataset_info.json           summary statistics

This format is directly loadable with:
    from datasets import load_dataset
    ds = load_dataset("audiofolder", data_dir="data/export")

And is compatible with Whisper fine-tuning scripts out of the box.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str | Path) -> List[Dict]:
    path = Path(manifest_path)
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_records(
    records: List[Dict],
    train_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict]]:
    """
    Deterministically split records into train / test sets.

    Records are sorted by id before splitting to ensure reproducibility.
    """
    records = sorted(records, key=lambda r: r.get("id", ""))
    n = len(records)
    n_test = max(1, round(n * test_ratio))
    n_train = n - n_test
    return {
        "train": records[:n_train],
        "test":  records[n_train:],
    }


# ── main export logic ─────────────────────────────────────────────────────────

def run_stage4(
    config: Dict,
    manifest_path: str | Path,
    run_id: Optional[str] = None,
) -> Path:
    """
    Export the reviewed dataset.

    Args:
        config:        Full pipeline config dict.
        manifest_path: Path to the Stage 2/3 manifest JSONL.
        run_id:        Optional run identifier (used in dataset_info.json).

    Returns:
        Path to the export root directory.
    """
    exp_cfg = config.get("export", {})
    output_dir = Path(exp_cfg.get("output_dir", "data/export"))
    train_ratio: float = exp_cfg.get("train_split", 0.9)
    test_ratio:  float = exp_cfg.get("test_split",  0.1)
    include_rejected: bool = exp_cfg.get("include_rejected", False)

    # ── 1. Load and filter manifest ───────────────────────────────────────────
    all_records = load_manifest(manifest_path)
    log.info("Loaded %d records from manifest.", len(all_records))

    if include_rejected:
        approved = [r for r in all_records if r["review_status"] != "pending"]
    else:
        approved = [r for r in all_records if r["review_status"] == "approved"]

    if not approved:
        log.warning(
            "No approved samples found in manifest.  "
            "Run the review UI (Stage 3) first and approve some samples."
        )
        return output_dir

    log.info(
        "%d approved samples will be exported (out of %d total).",
        len(approved),
        len(all_records),
    )

    # ── 2. Train/test split ───────────────────────────────────────────────────
    splits = _split_records(approved, train_ratio, test_ratio)

    # ── 3. Write each split ───────────────────────────────────────────────────
    split_stats: Dict[str, Dict] = {}
    for split_name, split_records in splits.items():
        split_dir  = output_dir / split_name
        audio_dir  = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        meta_path = split_dir / "metadata.jsonl"
        total_duration = 0.0

        with meta_path.open("w", encoding="utf-8") as fh:
            for rec in split_records:
                src_audio = Path(rec.get("audio_path", ""))
                if not src_audio.exists():
                    log.warning("Audio file missing, skipping: %s", src_audio)
                    continue

                dst_audio = audio_dir / src_audio.name
                shutil.copy2(src_audio, dst_audio)

                # Relative path from the split root (required by audiofolder)
                relative_audio = f"audio/{src_audio.name}"

                meta_entry = {
                    "file_name":     relative_audio,
                    "transcription": rec["text"],
                    # Extra metadata preserved for downstream use
                    "id":            rec["id"],
                    "domain":        rec.get("domain", ""),
                    "source":        rec.get("source", ""),
                    "engine":        rec.get("engine", ""),
                    "voice":         rec.get("voice", ""),
                    "duration_sec":  rec.get("duration_sec", 0.0),
                    "snr_db":        rec.get("snr_db", 0.0),
                }
                fh.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")
                total_duration += rec.get("duration_sec", 0.0)

        engine_counts = Counter(r.get("engine", "unknown") for r in split_records)
        domain_counts = Counter(r.get("domain", "unknown") for r in split_records)
        split_stats[split_name] = {
            "num_samples":       len(split_records),
            "total_duration_sec": round(total_duration, 2),
            "total_duration_min": round(total_duration / 60, 2),
            "engines":           dict(engine_counts),
            "domains":           dict(domain_counts),
        }
        log.info(
            "[%s] %d samples | %.1f min | engines: %s",
            split_name,
            len(split_records),
            total_duration / 60,
            dict(engine_counts),
        )

    # ── 4. Also write a rejected log (for audit purposes) ────────────────────
    rejected = [r for r in all_records if r["review_status"] == "rejected"]
    if rejected:
        rejected_path = output_dir / "rejected_log.jsonl"
        with rejected_path.open("w", encoding="utf-8") as fh:
            for rec in rejected:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Wrote %d rejected entries to %s", len(rejected), rejected_path)

    # ── 5. dataset_info.json ─────────────────────────────────────────────────
    all_approved_engines = Counter(r.get("engine", "") for r in approved)
    all_approved_domains = Counter(r.get("domain", "") for r in approved)
    total_approved_duration = sum(r.get("duration_sec", 0.0) for r in approved)

    dataset_info = {
        "run_id":             run_id or "unknown",
        "created_at":         datetime.now(timezone.utc).isoformat(),
        "manifest_source":    str(manifest_path),
        "total_samples":      len(approved),
        "total_duration_sec": round(total_approved_duration, 2),
        "total_duration_min": round(total_approved_duration / 60, 2),
        "splits":             split_stats,
        "engines":            dict(all_approved_engines),
        "domains":            dict(all_approved_domains),
        "total_rejected":     len(rejected),
        "total_pending":      sum(
            1 for r in all_records if r["review_status"] == "pending"
        ),
        "format": {
            "type":         "audiofolder",
            "audio_format": "wav",
            "sample_rate":  config.get("synthesis", {}).get("sample_rate", 24000),
            "transcription_field": "transcription",
            "load_with": (
                "from datasets import load_dataset\n"
                f"ds = load_dataset('audiofolder', data_dir='{output_dir}')"
            ),
        },
    }

    info_path = output_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset_info, fh, ensure_ascii=False, indent=2)

    log.info(
        "Stage 4 complete — dataset exported to %s\n"
        "  Total: %d samples | %.1f min\n"
        "  Load: from datasets import load_dataset; "
        "ds = load_dataset('audiofolder', data_dir='%s')",
        output_dir,
        len(approved),
        total_approved_duration / 60,
        output_dir,
    )
    return output_dir
