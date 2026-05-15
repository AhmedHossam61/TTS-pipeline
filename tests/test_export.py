"""Tests for pipeline/stage4_export.py"""
import json
import wave
import struct
from pathlib import Path

import pytest

from pipeline.stage4_export import run_stage4, load_manifest


def _make_dummy_wav(path: Path, duration_frames: int = 1000, sample_rate: int = 24000) -> None:
    """Write a minimal valid WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{duration_frames}h", *([0] * duration_frames)))


def _make_manifest(tmp_path: Path, records: list) -> Path:
    manifest = tmp_path / "manifest_test.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return manifest


class TestRunStage4:
    def _base_config(self, output_dir: Path) -> dict:
        return {
            "export": {
                "output_dir": str(output_dir),
                "train_split": 0.8,
                "test_split": 0.2,
                "include_rejected": False,
            },
            "synthesis": {"sample_rate": 24000},
        }

    def test_exports_approved_samples(self, tmp_path):
        audio_dir = tmp_path / "audio"
        wav1 = audio_dir / "p0001_edgetts_Salma.wav"
        wav2 = audio_dir / "p0002_edgetts_Shakir.wav"
        _make_dummy_wav(wav1)
        _make_dummy_wav(wav2)

        records = [
            {
                "id": "p0001_edgetts_Salma",
                "prompt_id": "p0001",
                "text": "إزيك النهارده؟",
                "domain": "conversation",
                "source": "seed",
                "engine": "edge-tts",
                "voice": "ar-EG-SalmaNeural",
                "audio_path": str(wav1),
                "duration_sec": 1.2,
                "snr_db": 25.0,
                "silence_ratio": 0.05,
                "quality_flags": [],
                "quality_passed": True,
                "review_status": "approved",
                "review_note": None,
            },
            {
                "id": "p0002_edgetts_Shakir",
                "prompt_id": "p0002",
                "text": "عامل إيه؟",
                "domain": "conversation",
                "source": "seed",
                "engine": "edge-tts",
                "voice": "ar-EG-ShakirNeural",
                "audio_path": str(wav2),
                "duration_sec": 0.9,
                "snr_db": 22.0,
                "silence_ratio": 0.03,
                "quality_flags": [],
                "quality_passed": True,
                "review_status": "approved",
                "review_note": None,
            },
        ]

        manifest = _make_manifest(tmp_path, records)
        export_dir = tmp_path / "export"
        config = self._base_config(export_dir)

        result = run_stage4(config, manifest, run_id="run_test")

        assert result.exists()
        # dataset_info.json must exist
        info_path = result / "dataset_info.json"
        assert info_path.exists()
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert info["total_samples"] == 2
        # At least one split must have metadata.jsonl
        assert any((result / split / "metadata.jsonl").exists()
                   for split in ("train", "test"))

    def test_skips_rejected_samples_by_default(self, tmp_path):
        audio_dir = tmp_path / "audio"
        wav = audio_dir / "p0001.wav"
        _make_dummy_wav(wav)

        records = [
            {
                "id": "p0001",
                "text": "مرحبا",
                "domain": "conversation",
                "source": "seed",
                "engine": "edge-tts",
                "voice": "ar-EG-SalmaNeural",
                "audio_path": str(wav),
                "duration_sec": 1.0,
                "snr_db": 20.0,
                "silence_ratio": 0.05,
                "quality_flags": [],
                "quality_passed": True,
                "review_status": "rejected",
                "review_note": "bad quality",
            },
        ]

        manifest = _make_manifest(tmp_path, records)
        export_dir = tmp_path / "export"
        config = self._base_config(export_dir)

        run_stage4(config, manifest, run_id="run_test")

        info_path = export_dir / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            assert info["total_samples"] == 0

    def test_metadata_jsonl_has_required_fields(self, tmp_path):
        audio_dir = tmp_path / "audio"
        wav = audio_dir / "p0001.wav"
        _make_dummy_wav(wav)

        records = [{
            "id": "p0001",
            "text": "مرحبا",
            "domain": "conversation",
            "source": "seed",
            "engine": "edge-tts",
            "voice": "ar-EG-SalmaNeural",
            "audio_path": str(wav),
            "duration_sec": 1.0,
            "snr_db": 20.0,
            "silence_ratio": 0.05,
            "quality_flags": [],
            "quality_passed": True,
            "review_status": "approved",
            "review_note": None,
        }]

        manifest = _make_manifest(tmp_path, records)
        export_dir = tmp_path / "export"
        config = self._base_config(export_dir)
        run_stage4(config, manifest, run_id="run_test")

        # Find the metadata.jsonl in whichever split has content
        for split in ("train", "test"):
            meta_path = export_dir / split / "metadata.jsonl"
            if meta_path.exists():
                content = meta_path.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                entry = json.loads(content.splitlines()[0])
                assert "file_name" in entry
                assert "transcription" in entry
                break


class TestLoadManifest:
    def test_loads_all_records(self, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            '{"id":"p1","text":"مرحبا"}\n{"id":"p2","text":"إزيك"}\n',
            encoding="utf-8",
        )
        records = load_manifest(manifest)
        assert len(records) == 2
        assert records[0]["id"] == "p1"

    def test_skips_blank_lines(self, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            '{"id":"p1","text":"مرحبا"}\n\n{"id":"p2","text":"إزيك"}\n',
            encoding="utf-8",
        )
        records = load_manifest(manifest)
        assert len(records) == 2
