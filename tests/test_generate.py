"""Tests for pipeline/stage1_generate.py"""
import json
import textwrap
from pathlib import Path

import pytest

from pipeline.stage1_generate import load_seeds, load_prompts, run_stage1


class TestLoadSeeds:
    def test_loads_sentences(self, tmp_path):
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text(
            textwrap.dedent("""\
                # ── Conversation ──
                إزيك النهارده؟
                عامل إيه يا صاحبي؟
                # comment line
                يلا بينا نروح نتغدى
            """),
            encoding="utf-8",
        )
        seeds = load_seeds(seed_file)
        texts = [s["text"] for s in seeds]
        assert "إزيك النهارده؟" in texts
        assert "عامل إيه يا صاحبي؟" in texts
        assert "يلا بينا نروح نتغدى" in texts
        # Comments must not appear
        assert not any("comment" in t for t in texts)

    def test_missing_file_returns_empty(self, tmp_path):
        seeds = load_seeds(tmp_path / "nonexistent.txt")
        assert seeds == []

    def test_empty_lines_skipped(self, tmp_path):
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("\n\nمرحبا\n\n", encoding="utf-8")
        seeds = load_seeds(seed_file)
        assert len(seeds) == 1

    def test_domain_extracted_from_section_header(self, tmp_path):
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text(
            "# ── Shopping ──\nبكام الكيلو؟\n",
            encoding="utf-8",
        )
        seeds = load_seeds(seed_file)
        assert seeds[0]["domain"] == "shopping"


class TestLoadPrompts:
    def test_roundtrip(self, tmp_path):
        manifest = tmp_path / "prompts.jsonl"
        records = [
            {"id": "p0001", "text": "مرحبا", "domain": "conversation", "source": "seed"},
            {"id": "p0002", "text": "إزيك؟", "domain": "conversation", "source": "gemini"},
        ]
        with manifest.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        loaded = load_prompts(manifest)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "p0001"
        assert loaded[1]["source"] == "gemini"


class TestRunStage1:
    def test_seeds_only_no_gemini(self, tmp_path, monkeypatch):
        # Disable Gemini expansion so the test is offline
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("مرحبا\nإزيك النهارده؟\n", encoding="utf-8")

        config = {
            "generation": {
                "num_prompts": 10,
                "expand_with_gemini": False,
                "seed_file": str(seed_file),
                "output_dir": str(tmp_path / "prompts"),
                "strip_diacritics": True,
                "domains": ["conversation"],
                "gemini_model": "gemini-1.5-flash",
                "gemini_sentences_per_call": 5,
            }
        }

        manifest = run_stage1(config, "run_test")
        assert manifest.exists()

        prompts = load_prompts(manifest)
        assert len(prompts) == 2
        # IDs should be sequential
        assert prompts[0]["id"] == "p0001"
        assert prompts[1]["id"] == "p0002"

    def test_deduplication(self, tmp_path):
        seed_file = tmp_path / "seeds.txt"
        # Duplicate sentence
        seed_file.write_text("مرحبا\nمرحبا\nإزيك؟\n", encoding="utf-8")

        config = {
            "generation": {
                "num_prompts": 10,
                "expand_with_gemini": False,
                "seed_file": str(seed_file),
                "output_dir": str(tmp_path / "prompts"),
                "strip_diacritics": True,
                "domains": [],
                "gemini_model": "gemini-1.5-flash",
                "gemini_sentences_per_call": 5,
            }
        }

        manifest = run_stage1(config, "run_dedup")
        prompts = load_prompts(manifest)
        texts = [p["text"] for p in prompts]
        assert len(texts) == len(set(texts)), "Duplicates were not removed"
