"""
pipeline/stage1_generate.py
────────────────────────────
Stage 1 — Prompt Generation.

Flow:
  1. Load seed sentences from a plain-text file.
  2. Optionally expand the corpus by calling the Gemini API with
     domain-specific prompts that ask for authentic Egyptian Arabic sentences.
  3. Normalise every sentence (strip diacritics, collapse whitespace, etc.).
  4. Deduplicate, then write a JSONL prompt manifest.

Output manifest schema (one JSON object per line):
  {
    "id":         "p0001",
    "text":       "إزيك النهارده؟",
    "domain":     "conversation",
    "source":     "seed" | "gemini",
    "char_count": 15,
    "created_at": "2026-05-15T10:00:00+00:00"
  }
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from utils.arabic_utils import is_arabic_text, normalize_text

log = logging.getLogger(__name__)

# ── Gemini helper ─────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = (
    "You are an expert in Egyptian Arabic dialect (اللهجة المصرية العامية). "
    "Generate authentic Egyptian colloquial sentences — NOT Modern Standard Arabic (فصحى). "
    "Use typical Egyptian expressions: يعني، بقى، اهو، اشمعنى، والنبي، إزيك، etc. "
    "Code-switching with English words is acceptable and common in Egyptian speech. "
    "Output ONLY the sentences, one per line, with no numbering, bullet points, or extra text."
)


def _build_gemini_prompt(domain: str, count: int) -> str:
    n_short = max(1, count // 3)
    n_medium = max(1, count // 3)
    n_long = count - n_short - n_medium
    return (
        f"Generate exactly {count} Egyptian Arabic sentences for the domain: **{domain}**.\n"
        f"Mix sentence lengths:\n"
        f"  - {n_short} short sentences (3–7 words)\n"
        f"  - {n_medium} medium sentences (8–15 words)\n"
        f"  - {n_long} longer sentences (16–25 words)\n"
        f"Make them sound like natural spoken Egyptian Arabic."
    )


def _call_gemini(
    model,
    domain: str,
    count: int,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> List[str]:
    """Call Gemini and return a list of raw sentence strings."""
    prompt = _build_gemini_prompt(domain, count)
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            raw = response.text or ""
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            # Drop any line that looks like a numbering artefact
            lines = [re.sub(r"^\d+[.\)]\s*", "", l) for l in lines]
            lines = [l for l in lines if is_arabic_text(l)]
            return lines
        except Exception as exc:
            log.warning(
                "Gemini call failed (attempt %d/%d) for domain '%s': %s",
                attempt, max_retries, domain, exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    return []


# ── Seed loader ───────────────────────────────────────────────────────────────

def load_seeds(seed_file: str | Path) -> List[Dict]:
    """
    Load sentences from a plain-text seed file.

    Lines starting with # are treated as comments.
    Empty lines are skipped.
    Domain tags are NOT extracted — all seeds are labelled domain='general'.
    """
    path = Path(seed_file)
    if not path.exists():
        log.warning("Seed file not found: %s", path)
        return []

    sentences = []
    current_domain = "general"
    domain_comment_re = re.compile(r"^#\s*[──-]+\s*(.+?)\s*[──-]*\s*$")

    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Try to extract domain from section header comments
                m = domain_comment_re.match(line)
                if m:
                    # Clean up the domain label → snake_case
                    label = m.group(1).lower()
                    label = re.sub(r"\s+/\s+|\s+", "_", label)
                    label = re.sub(r"[^\w]", "", label)
                    current_domain = label or "general"
                continue
            sentences.append({"text": line, "domain": current_domain, "source": "seed"})

    log.info("Loaded %d seed sentences from %s", len(sentences), path)
    return sentences


# ── Main stage entry point ────────────────────────────────────────────────────

def run_stage1(config: Dict, run_id: str) -> Path:
    """
    Execute Stage 1 and write a JSONL prompt manifest.

    Args:
        config: The full pipeline config dict (as loaded from config.yaml).
        run_id: Unique identifier for this pipeline run.

    Returns:
        Path to the written JSONL manifest file.
    """
    gen_cfg = config["generation"]
    output_dir = Path(gen_cfg.get("output_dir", "data/prompts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    strip_diacritics_flag: bool = gen_cfg.get("strip_diacritics", True)

    # ── 1. Load seeds ─────────────────────────────────────────────────────────
    seeds = load_seeds(gen_cfg["seed_file"])
    num_prompts: int = gen_cfg.get("num_prompts", 200)
    
    # If Gemini expansion is disabled, sample only num_prompts seeds
    if not gen_cfg.get("expand_with_gemini", True):
        if len(seeds) > num_prompts:
            seeds = random.sample(seeds, num_prompts)
            log.info("Sampled %d seeds from %d available", len(seeds), len(load_seeds(gen_cfg["seed_file"])))

    # ── 2. Expand with Gemini ─────────────────────────────────────────────────
    gemini_sentences: List[Dict] = []

    if gen_cfg.get("expand_with_gemini", True):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            log.warning("GEMINI_API_KEY not set — skipping Gemini expansion.")
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=gen_cfg.get("gemini_model", "gemini-1.5-flash"),
                    system_instruction=_SYSTEM_INSTRUCTION,
                )

                domains: List[str] = gen_cfg.get("domains", ["conversation"])
                per_call: int = gen_cfg.get("gemini_sentences_per_call", 20)
                # Distribute target count across domains
                per_domain = max(1, num_prompts // len(domains))

                for domain in domains:
                    calls_needed = max(1, per_domain // per_call)
                    for _ in range(calls_needed):
                        lines = _call_gemini(model, domain, per_call)
                        for line in lines:
                            gemini_sentences.append(
                                {"text": line, "domain": domain, "source": "gemini"}
                            )
                        log.info(
                            "Gemini: +%d sentences for domain '%s'", len(lines), domain
                        )
                        time.sleep(1.0)  # be polite to the API

            except ImportError:
                log.warning(
                    "google-generativeai not installed — "
                    "run: pip install google-generativeai"
                )
            except Exception as exc:
                log.error("Gemini expansion failed: %s", exc)

    # ── 3. Merge, normalise, deduplicate ──────────────────────────────────────
    all_raw = seeds + gemini_sentences
    seen: set[str] = set()
    prompts: List[Dict] = []

    for idx, item in enumerate(all_raw, start=1):
        normalised = normalize_text(item["text"], strip_diacritics_flag)
        if not normalised or not is_arabic_text(normalised):
            log.debug("Skipping non-Arabic/empty sentence: %r", item["text"][:60])
            continue
        if normalised in seen:
            log.debug("Duplicate skipped: %r", normalised[:60])
            continue
        seen.add(normalised)
        prompts.append(
            {
                "id": f"p{idx:04d}",
                "text": normalised,
                "domain": item["domain"],
                "source": item["source"],
                "char_count": len(normalised),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Re-number after deduplication
    for idx, p in enumerate(prompts, start=1):
        p["id"] = f"p{idx:04d}"

    # ── 4. Write manifest ─────────────────────────────────────────────────────
    manifest_path = output_dir / f"prompts_{run_id}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info(
        "Stage 1 complete — %d prompts written to %s "
        "(%d from seeds, %d from Gemini)",
        len(prompts),
        manifest_path,
        sum(1 for p in prompts if p["source"] == "seed"),
        sum(1 for p in prompts if p["source"] == "gemini"),
    )
    return manifest_path


# ── Utility: load a prompts manifest ─────────────────────────────────────────

def load_prompts(manifest_path: str | Path) -> List[Dict]:
    """Load all prompts from a JSONL manifest file."""
    path = Path(manifest_path)
    prompts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts
