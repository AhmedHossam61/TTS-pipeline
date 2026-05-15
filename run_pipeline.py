"""
run_pipeline.py
───────────────
End-to-end pipeline runner with a simple CLI.

Usage
─────
  # Run all four stages in sequence
  python run_pipeline.py --stage all

  # Run individual stages
  python run_pipeline.py --stage generate
  python run_pipeline.py --stage synthesize
  python run_pipeline.py --stage review
  python run_pipeline.py --stage export

  # Resume using an existing run ID (skips already-completed synthesis jobs)
  python run_pipeline.py --stage all --run-id run_20260515_100000

  # Use a different config file
  python run_pipeline.py --stage all --config config/my_config.yaml

  # Override the prompts manifest for synthesis (skip Stage 1)
  python run_pipeline.py --stage synthesize --prompts data/prompts/prompts_run_xxx.jsonl

  # Override the manifest for review/export (skip earlier stages)
  python run_pipeline.py --stage review   --manifest data/manifests/manifest_run_xxx.jsonl
  python run_pipeline.py --stage export   --manifest data/manifests/manifest_run_xxx.jsonl
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _make_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_latest_manifest(manifest_dir: str | Path) -> Path | None:
    """Return the most recently modified manifest JSONL in *manifest_dir*."""
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.exists():
        return None
    candidates = sorted(
        manifest_dir.glob("manifest_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_latest_prompts(prompts_dir: str | Path) -> Path | None:
    """Return the most recently modified prompts JSONL in *prompts_dir*."""
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.exists():
        return None
    candidates = sorted(
        prompts_dir.glob("prompts_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ── stage runners ─────────────────────────────────────────────────────────────

def stage_generate(config: dict, run_id: str) -> Path:
    from pipeline.stage1_generate import run_stage1
    log.info("=" * 60)
    log.info("STAGE 1 — Prompt Generation  (run_id=%s)", run_id)
    log.info("=" * 60)
    return run_stage1(config, run_id)


def stage_synthesize(config: dict, run_id: str, prompts_manifest: Path) -> Path:
    from pipeline.stage2_synthesize import run_stage2
    log.info("=" * 60)
    log.info("STAGE 2 — TTS Synthesis  (run_id=%s)", run_id)
    log.info("Prompts manifest: %s", prompts_manifest)
    log.info("=" * 60)
    return run_stage2(config, run_id, prompts_manifest)


def stage_review(config: dict, manifest_path: Path) -> None:
    from pipeline.stage3_review import run_stage3
    log.info("=" * 60)
    log.info("STAGE 3 — Review UI")
    log.info("Manifest: %s", manifest_path)
    log.info("=" * 60)
    run_stage3(config, manifest_path)


def stage_export(config: dict, manifest_path: Path, run_id: str) -> Path:
    from pipeline.stage4_export import run_stage4
    log.info("=" * 60)
    log.info("STAGE 4 — Export  (run_id=%s)", run_id)
    log.info("Manifest: %s", manifest_path)
    log.info("=" * 60)
    return run_stage4(config, manifest_path, run_id)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Speech Data Pipeline (SSDP) — end-to-end runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "synthesize", "review", "export"],
        default="all",
        help="Which stage(s) to run (default: all).",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing run ID to resume interrupted synthesis.",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to prompts JSONL (skips Stage 1 when provided).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to synthesis manifest JSONL (skips Stage 1+2 when provided).",
    )
    args = parser.parse_args()

    # ── load .env and config ──────────────────────────────────────────────────
    load_dotenv(Path("config") / ".env", override=False)
    load_dotenv(override=False)  # also check project root .env

    if not Path(args.config).exists():
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = _load_config(args.config)

    # ── resolve run ID ────────────────────────────────────────────────────────
    cfg_run_id = config.get("pipeline", {}).get("run_id")
    run_id: str = args.run_id or cfg_run_id or _make_run_id()
    log.info("Run ID: %s", run_id)

    # ── convenience paths ─────────────────────────────────────────────────────
    syn_cfg  = config.get("synthesis", {})
    gen_cfg  = config.get("generation", {})
    prompts_dir  = Path(gen_cfg.get("output_dir",  "data/prompts"))
    manifest_dir = Path(syn_cfg.get("manifest_dir", "data/manifests"))

    # ── dispatch ──────────────────────────────────────────────────────────────
    stage = args.stage

    if stage in ("all", "generate"):
        prompts_manifest = stage_generate(config, run_id)
    else:
        # Resolve prompts manifest from --prompts flag or latest file on disk
        if args.prompts:
            prompts_manifest = Path(args.prompts)
        else:
            prompts_manifest = _resolve_latest_prompts(prompts_dir)
        if prompts_manifest is None and stage == "synthesize":
            log.error(
                "No prompts manifest found.  "
                "Run Stage 1 first or pass --prompts <path>."
            )
            sys.exit(1)

    if stage in ("all", "synthesize"):
        synthesis_manifest = stage_synthesize(config, run_id, prompts_manifest)
    else:
        # Resolve manifest from --manifest flag or latest file on disk
        if args.manifest:
            synthesis_manifest = Path(args.manifest)
        else:
            synthesis_manifest = _resolve_latest_manifest(manifest_dir)
        if synthesis_manifest is None and stage in ("review", "export"):
            log.error(
                "No synthesis manifest found.  "
                "Run Stage 2 first or pass --manifest <path>."
            )
            sys.exit(1)

    if stage == "review":
        stage_review(config, synthesis_manifest)
        return  # Gradio app blocks until closed

    if stage in ("all", "export"):
        if stage == "all":
            # After a full run, prompt the user to review before exporting
            log.info(
                "\n"
                "  ┌─────────────────────────────────────────────────────┐\n"
                "  │  Stage 2 complete.                                   │\n"
                "  │  Run Stage 3 to review samples before exporting:     │\n"
                "  │    python run_pipeline.py --stage review             │\n"
                "  │                                                      │\n"
                "  │  Or export immediately (all quality-passed samples): │\n"
                "  │    python run_pipeline.py --stage export             │\n"
                "  └─────────────────────────────────────────────────────┘\n"
            )
            return

        export_dir = stage_export(config, synthesis_manifest, run_id)
        log.info("Dataset exported to: %s", export_dir)


if __name__ == "__main__":
    main()
