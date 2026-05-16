"""
auto_model_evals.py
-------------------
Automatic STT evaluation utility for synthesized audio manifests.

Primary use:
- Evaluate Google Vertex/Cloud Speech-to-Text output against manifest text.

Future-proof use:
- Evaluate any fine-tuned STT model by passing a predictions JSONL file
  (id -> predicted_text), without changing this script.

Examples
--------
# 1) Evaluate with Vertex STT
python auto_model_evals.py \
  --manifest data/manifests/manifest_run_20260516_094139.jsonl \
  --backend vertex \
  --language-code ar-EG

# 2) Evaluate a future fine-tuned model from precomputed predictions
python auto_model_evals.py \
  --manifest data/manifests/manifest_run_20260516_094139.jsonl \
  --backend predictions \
  --predictions data/evals/my_model_predictions.jsonl

Predictions JSONL schema
------------------------
One object per line:
{"id": "run_xxx_p0001_edge-tts_ar_EG_SalmaNeural", "predicted_text": "..."}
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.arabic_utils import normalize_text

log = logging.getLogger(__name__)


@dataclass
class EvalSample:
    sample_id: str
    reference: str
    predicted: str
    audio_path: str
    engine: str
    voice: str
    domain: str
    source: str


def load_manifest(manifest_path: str | Path) -> List[Dict]:
    path = Path(manifest_path)
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _levenshtein(seq_a: List[str], seq_b: List[str]) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        curr = [i]
        for j, b in enumerate(seq_b, start=1):
            cost = 0 if a == b else 1
            curr.append(
                min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _levenshtein(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars))


class STTBackend:
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class VertexSTTBackend(STTBackend):
    """Google Cloud Speech-to-Text backend."""

    def __init__(
        self,
        language_code: str = "ar-EG",
        model: Optional[str] = None,
        use_enhanced: bool = False,
        sample_rate_hz: Optional[int] = None,
    ) -> None:
        try:
            from google.cloud import speech  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-speech is required for backend=vertex. "
                "Install with: pip install google-cloud-speech"
            ) from exc

        self._speech = speech
        self._client = speech.SpeechClient()
        self._language_code = language_code
        self._model = model
        self._use_enhanced = use_enhanced
        self._sample_rate_hz = sample_rate_hz

    def transcribe(self, audio_path: str) -> str:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        content = p.read_bytes()
        audio = self._speech.RecognitionAudio(content=content)

        cfg_kwargs = {
            "language_code": self._language_code,
            "enable_automatic_punctuation": True,
        }
        if self._model:
            cfg_kwargs["model"] = self._model
        if self._use_enhanced:
            cfg_kwargs["use_enhanced"] = True
        if self._sample_rate_hz:
            cfg_kwargs["sample_rate_hertz"] = self._sample_rate_hz

        config = self._speech.RecognitionConfig(**cfg_kwargs)
        response = self._client.recognize(config=config, audio=audio)

        parts: List[str] = []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
        return " ".join(parts).strip()


class PredictionsFileBackend(STTBackend):
    """Backend that reads predicted text from JSONL (for future custom models)."""

    def __init__(self, predictions_path: str | Path) -> None:
        self._predictions = self._load_predictions(predictions_path)

    @staticmethod
    def _load_predictions(predictions_path: str | Path) -> Dict[str, str]:
        path = Path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")

        out: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("id", "")).strip()
                pred = str(
                    row.get("predicted_text")
                    or row.get("prediction")
                    or row.get("text")
                    or ""
                ).strip()
                if not sid:
                    raise ValueError(f"Missing id at line {line_no} in {path}")
                out[sid] = pred
        return out

    def transcribe(self, audio_path: str) -> str:
        raise RuntimeError("Use transcribe_by_id for predictions backend")

    def transcribe_by_id(self, sample_id: str) -> str:
        return self._predictions.get(sample_id, "")


def _build_backend(args: argparse.Namespace) -> STTBackend:
    if args.backend == "vertex":
        return VertexSTTBackend(
            language_code=args.language_code,
            model=args.model,
            use_enhanced=args.use_enhanced,
            sample_rate_hz=args.sample_rate_hz,
        )
    if args.backend == "predictions":
        if not args.predictions:
            raise ValueError("--predictions is required when --backend predictions")
        return PredictionsFileBackend(args.predictions)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _normalize_for_eval(text: str, strip_diacritics_flag: bool = True) -> str:
    return normalize_text(text or "", strip_diacritics_flag=strip_diacritics_flag)


def _compute_global_metrics(samples: List[EvalSample]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "wer_mean": 0.0, "cer_mean": 0.0}

    wers = [wer(s.reference, s.predicted) for s in samples]
    cers = [cer(s.reference, s.predicted) for s in samples]
    return {
        "count": len(samples),
        "wer_mean": sum(wers) / len(wers),
        "cer_mean": sum(cers) / len(cers),
    }


def _group_metrics(samples: List[EvalSample], key_name: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[EvalSample]] = defaultdict(list)
    for s in samples:
        key = getattr(s, key_name) or "unknown"
        groups[key].append(s)

    out: Dict[str, Dict[str, float]] = {}
    for key, bucket in groups.items():
        out[key] = _compute_global_metrics(bucket)
    return out


def evaluate(args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    records = load_manifest(args.manifest)
    backend = _build_backend(args)

    filtered: List[Dict] = []
    for r in records:
        if args.only_approved and r.get("review_status") != "approved":
            continue
        filtered.append(r)

    samples: List[EvalSample] = []
    per_sample_rows: List[Dict] = []

    for rec in filtered:
        sid = rec.get("id")
        ref_raw = rec.get("text", "")
        audio_path = rec.get("audio_path", "")

        if not sid or not ref_raw or not audio_path:
            continue

        if args.backend == "predictions":
            pred_raw = backend.transcribe_by_id(sid)  # type: ignore[attr-defined]
        else:
            pred_raw = backend.transcribe(audio_path)

        ref_norm = _normalize_for_eval(ref_raw, strip_diacritics_flag=args.strip_diacritics)
        pred_norm = _normalize_for_eval(pred_raw, strip_diacritics_flag=args.strip_diacritics)

        sample = EvalSample(
            sample_id=sid,
            reference=ref_norm,
            predicted=pred_norm,
            audio_path=audio_path,
            engine=rec.get("engine", ""),
            voice=rec.get("voice", ""),
            domain=rec.get("domain", ""),
            source=rec.get("source", ""),
        )
        samples.append(sample)

        s_wer = wer(ref_norm, pred_norm)
        s_cer = cer(ref_norm, pred_norm)
        per_sample_rows.append(
            {
                "id": sid,
                "engine": sample.engine,
                "voice": sample.voice,
                "domain": sample.domain,
                "source": sample.source,
                "audio_path": audio_path,
                "reference_text": ref_norm,
                "predicted_text": pred_norm,
                "wer": s_wer,
                "cer": s_cer,
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(args.manifest)),
        "backend": args.backend,
        "language_code": args.language_code,
        "model": args.model,
        "only_approved": args.only_approved,
        "strip_diacritics": args.strip_diacritics,
        "global": _compute_global_metrics(samples),
        "by_engine": _group_metrics(samples, "engine"),
        "by_voice": _group_metrics(samples, "voice"),
        "by_domain": _group_metrics(samples, "domain"),
        "by_source": _group_metrics(samples, "source"),
    }

    return summary, per_sample_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic STT evaluator for synthesized Arabic speech manifests"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to synthesis manifest JSONL",
    )
    parser.add_argument(
        "--backend",
        choices=["vertex", "predictions"],
        default="vertex",
        help="STT backend: vertex API or local predictions file",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSONL path (required for backend=predictions)",
    )
    parser.add_argument(
        "--language-code",
        default="ar-EG",
        help="STT language code (default: ar-EG)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Vertex Speech model name",
    )
    parser.add_argument(
        "--use-enhanced",
        action="store_true",
        help="Enable enhanced model in Vertex Speech when available",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=None,
        help="Optional override for sample rate in STT config",
    )
    parser.add_argument(
        "--only-approved",
        action="store_true",
        help="Evaluate only records with review_status=approved",
    )
    parser.add_argument(
        "--strip-diacritics",
        action="store_true",
        help="Normalize by stripping Arabic diacritics before scoring",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evals",
        help="Directory where summary JSON and per-sample JSONL are saved",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    summary, per_sample = evaluate(args)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    summary_path = out_dir / f"eval_summary_{ts}.json"
    rows_path = out_dir / f"eval_samples_{ts}.jsonl"

    save_json(summary_path, summary)
    save_jsonl(rows_path, per_sample)

    g = summary["global"]
    log.info(
        "Evaluation complete. count=%d | WER=%.4f | CER=%.4f",
        int(g["count"]),
        float(g["wer_mean"]),
        float(g["cer_mean"]),
    )
    log.info("Summary saved to: %s", summary_path)
    log.info("Per-sample scores saved to: %s", rows_path)


if __name__ == "__main__":
    main()
