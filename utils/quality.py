"""
utils/quality.py
────────────────
Automated quality signals for synthesized audio clips.

Checks performed:
  - Duration (too short or too long)
  - SNR estimate (signal-to-noise ratio via energy percentile method)
  - Silence ratio (leading + trailing silence as a fraction of total duration)

None of these checks auto-reject a sample — they produce flags that are shown
in the Gradio review UI and stored in the manifest for human review.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

# librosa import is deferred so the module can be imported without it
# (tests that don't touch audio can still import this module)
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIBROSA_AVAILABLE = False
    log.warning("librosa not installed — quality checks will be skipped.")


@dataclass
class QualityReport:
    """Result of running quality checks on a single audio clip."""

    audio_path: str
    duration_sec: float = 0.0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    flags: List[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "snr_db": round(self.snr_db, 2),
            "silence_ratio": round(self.silence_ratio, 3),
            "quality_flags": self.flags,
            "quality_passed": self.passed,
        }


# ── individual signal functions ───────────────────────────────────────────────

def get_duration(audio_path: str | Path, sr: int = 22050) -> float:
    """Return duration in seconds."""
    if not _LIBROSA_AVAILABLE:
        return 0.0
    try:
        return librosa.get_duration(path=str(audio_path))
    except Exception as exc:
        log.debug("Duration check failed for %s: %s", audio_path, exc)
        return 0.0


def estimate_snr(audio_path: str | Path, sr: int = 22050) -> float:
    """
    Estimate SNR in dB using an energy-percentile approach.

    The 5th-percentile frame energy is treated as the noise floor;
    the 95th-percentile is treated as the signal.  Returns the ratio
    in dB.  Returns 0.0 on any error.
    """
    if not _LIBROSA_AVAILABLE:
        return 0.0
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        if y.size == 0:
            return 0.0
        frame_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        frame_energy = frame_energy[frame_energy > 0]
        if frame_energy.size < 4:
            return 0.0
        noise = float(np.percentile(frame_energy, 5))
        signal = float(np.percentile(frame_energy, 95))
        if noise <= 0:
            return 60.0  # essentially silence-free — very clean
        return float(20 * np.log10(signal / noise))
    except Exception as exc:
        log.debug("SNR estimate failed for %s: %s", audio_path, exc)
        return 0.0


def estimate_silence_ratio(
    audio_path: str | Path, sr: int = 22050, top_db: float = 30.0
) -> float:
    """
    Return the fraction of the clip that is leading or trailing silence.

    Uses librosa.effects.trim to find the non-silent region.

    Returns:
        Float in [0, 1].  A value near 1 means the clip is almost all silence.
    """
    if not _LIBROSA_AVAILABLE:
        return 0.0
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        if y.size == 0:
            return 1.0
        _, (start_sample, end_sample) = librosa.effects.trim(y, top_db=top_db)
        total = len(y)
        non_silent = end_sample - start_sample
        silent = total - non_silent
        return float(silent / total) if total > 0 else 1.0
    except Exception as exc:
        log.debug("Silence ratio failed for %s: %s", audio_path, exc)
        return 0.0


# ── top-level checker ─────────────────────────────────────────────────────────

def quality_check(audio_path: str | Path, config: Dict) -> QualityReport:
    """
    Run all quality checks on *audio_path* against the thresholds in *config*.

    Args:
        audio_path: Path to the WAV file.
        config: The ``quality`` sub-dict from config.yaml, e.g.:
            {
                "min_duration_sec": 0.5,
                "max_duration_sec": 30.0,
                "min_snr_db": 10.0,
                "max_silence_ratio": 0.35,
            }

    Returns:
        A QualityReport with flags for each violated threshold.
    """
    report = QualityReport(audio_path=str(audio_path))

    if not Path(audio_path).exists():
        report.flags.append("file_missing")
        report.passed = False
        return report

    report.duration_sec = get_duration(audio_path)
    report.snr_db = estimate_snr(audio_path)
    report.silence_ratio = estimate_silence_ratio(audio_path)

    min_dur = config.get("min_duration_sec", 0.5)
    max_dur = config.get("max_duration_sec", 30.0)
    min_snr = config.get("min_snr_db", 10.0)
    max_sil = config.get("max_silence_ratio", 0.35)

    if report.duration_sec < min_dur:
        report.flags.append(f"too_short ({report.duration_sec:.2f}s < {min_dur}s)")
    if report.duration_sec > max_dur:
        report.flags.append(f"too_long ({report.duration_sec:.1f}s > {max_dur}s)")
    if report.snr_db < min_snr:
        report.flags.append(f"low_snr ({report.snr_db:.1f} dB < {min_snr} dB)")
    if report.silence_ratio > max_sil:
        report.flags.append(
            f"excessive_silence ({report.silence_ratio:.0%} > {max_sil:.0%})"
        )

    report.passed = len(report.flags) == 0
    return report
