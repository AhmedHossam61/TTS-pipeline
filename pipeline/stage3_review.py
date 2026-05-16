"""
pipeline/stage3_review.py
─────────────────────────
Stage 3 — Gradio-based Review UI.

Loads a synthesis manifest and lets a human reviewer:
  • Listen to each (text, audio) pair.
  • Approve ✅ / Reject ❌ / Skip ⏭ each sample.
  • Add a free-text note (e.g. "wrong pronunciation", "clipping").
  • See automated quality flags (SNR, duration, silence) per sample.
  • Use keyboard shortcuts: A=approve, R=reject, N=next sample.
  • Bulk reject pending clips where quality_passed=false.

Decisions are persisted to the manifest JSONL file in real time so that
progress survives browser refreshes or app restarts.

Usage:
  python run_pipeline.py --stage review
  # or directly:
  python -m pipeline.stage3_review --manifest data/manifests/manifest_run_xxx.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


# ── Manifest I/O helpers ──────────────────────────────────────────────────────

def load_manifest(manifest_path: str | Path) -> List[Dict]:
    path = Path(manifest_path)
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_manifest(manifest_path: str | Path, records: List[Dict]) -> None:
    path = Path(manifest_path)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _update_record(
    records: List[Dict],
    idx: int,
    status: str,
    note: str,
    manifest_path: str | Path,
) -> None:
    """Mutate the record at *idx* and persist the manifest to disk."""
    records[idx]["review_status"] = status
    records[idx]["review_note"] = note.strip() if note else None
    save_manifest(manifest_path, records)


# ── Gradio app builder ────────────────────────────────────────────────────────

def build_app(manifest_path: str | Path):
    """Build and return the Gradio Blocks app."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed.  Run: pip install gradio"
        ) from exc

    records = load_manifest(manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    total = len(records)
    state_idx = [0]

    def _stats_text() -> str:
        approved = sum(1 for r in records if r["review_status"] == "approved")
        rejected = sum(1 for r in records if r["review_status"] == "rejected")
        pending = sum(1 for r in records if r["review_status"] == "pending")
        return (
            f"**Total:** {total}  |  "
            f"✅ Approved: {approved}  |  "
            f"❌ Rejected: {rejected}  |  "
            f"⏳ Pending: {pending}"
        )

    def _flags_html(rec: Dict) -> str:
        flags = rec.get("quality_flags", [])
        if not flags:
            return "<span style='color:green'>✔ All quality checks passed</span>"
        parts = "".join(
            f"<span style='background:#ffe0e0;border-radius:4px;padding:2px 6px;"
            f"margin:2px;display:inline-block'>⚠ {f}</span>"
            for f in flags
        )
        return parts

    def _load_sample(idx: int):
        rec = records[idx]
        audio = rec.get("audio_path", "")
        audio_val = audio if audio and Path(audio).exists() else None
        note_val = rec.get("review_note") or ""
        progress = f"Sample {idx + 1} / {total}  —  ID: {rec['id']}"
        status_badge = {
            "approved": "✅ Approved",
            "rejected": "❌ Rejected",
            "skipped": "⏭ Skipped",
            "pending": "⏳ Pending",
        }.get(rec["review_status"], rec["review_status"])

        meta = (
            f"**Domain:** {rec.get('domain', '—')}  |  "
            f"**Engine:** {rec.get('engine', '—')}  |  "
            f"**Voice:** {rec.get('voice', '—')}  |  "
            f"**Duration:** {rec.get('duration_sec', 0):.2f}s  |  "
            f"**SNR:** {rec.get('snr_db', 0):.1f} dB  |  "
            f"**Status:** {status_badge}"
        )
        return (
            rec["text"],
            audio_val,
            _flags_html(rec),
            meta,
            note_val,
            progress,
            _stats_text(),
        )

    def _navigate(delta: int):
        idx = max(0, min(total - 1, state_idx[0] + delta))
        state_idx[0] = idx
        return _load_sample(idx)

    def _jump_to_next_pending():
        for i in range(total):
            if records[i]["review_status"] == "pending":
                state_idx[0] = i
                return _load_sample(i)
        return _load_sample(state_idx[0])

    def _action(status: str, note: str):
        idx = state_idx[0]
        _update_record(records, idx, status, note, manifest_path)
        return _jump_to_next_pending()

    def _batch_reject_failed_quality():
        changed = 0
        for rec in records:
            if rec.get("review_status") != "pending":
                continue
            if rec.get("quality_passed", True):
                continue
            rec["review_status"] = "rejected"
            if not rec.get("review_note"):
                rec["review_note"] = "auto_reject: quality_passed=false"
            changed += 1

        if changed:
            save_manifest(manifest_path, records)

        return _jump_to_next_pending()

    _keyboard_js = """
    () => {
      function findBtn(text) {
        return Array.from(document.querySelectorAll('button'))
          .find(b => b.textContent.trim() === text);
      }
      document.addEventListener('keydown', (e) => {
        const tag = (document.activeElement?.tagName || '').toUpperCase();
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        const key = e.key.toLowerCase();
        if (key === 'a') { findBtn('\u2705 Approve')?.click(); e.preventDefault(); }
        else if (key === 'r') { findBtn('\u274c Reject')?.click(); e.preventDefault(); }
        else if (key === 'n') { findBtn('\u25b6 Next')?.click(); e.preventDefault(); }
      });
    }
    """

    with gr.Blocks(
        title="TTS Pipeline — Review UI",
        css="""
        #arabic-text {
            font-size: 2rem;
            direction: rtl;
            text-align: right;
            font-family: 'Amiri', 'Noto Naskh Arabic', serif;
            background: #f9f5ee;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        """,
        js=_keyboard_js,
    ) as demo:
        gr.Markdown("# 🎙 TTS Pipeline — Review UI")
        gr.Markdown("**Shortcuts:** `A` = Approve, `R` = Reject, `N` = Next")
        stats_md = gr.Markdown(_stats_text())

        with gr.Row():
            progress_txt = gr.Textbox(label="Progress", interactive=False, scale=3)
            jump_btn = gr.Button("⏩ Jump to next pending", scale=1)

        batch_reject_btn = gr.Button(
            "⚡ Batch reject pending clips with quality_passed = false",
            variant="secondary",
        )

        arabic_txt = gr.Textbox(
            label="Arabic Text",
            elem_id="arabic-text",
            interactive=False,
            lines=3,
        )
        audio_player = gr.Audio(label="Synthesized Audio", type="filepath")

        with gr.Row():
            meta_md = gr.Markdown()

        quality_html = gr.HTML(label="Quality Flags")

        note_box = gr.Textbox(
            label="Review Note (optional)",
            placeholder="e.g. wrong pronunciation, clipping, background noise …",
            lines=2,
        )

        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary", elem_id="approve-btn")
            reject_btn = gr.Button("❌ Reject", variant="stop", elem_id="reject-btn")
            skip_btn = gr.Button("⏭ Skip")

        with gr.Row():
            prev_btn = gr.Button("◀ Previous")
            next_btn = gr.Button("▶ Next", elem_id="next-btn")

        outputs = [arabic_txt, audio_player, quality_html, meta_md, note_box, progress_txt, stats_md]

        approve_btn.click(
            fn=lambda note: _action("approved", note),
            inputs=[note_box],
            outputs=outputs,
        )
        reject_btn.click(
            fn=lambda note: _action("rejected", note),
            inputs=[note_box],
            outputs=outputs,
        )
        skip_btn.click(
            fn=lambda note: _action("skipped", note),
            inputs=[note_box],
            outputs=outputs,
        )
        prev_btn.click(fn=lambda: _navigate(-1), outputs=outputs)
        next_btn.click(fn=lambda: _navigate(+1), outputs=outputs)
        jump_btn.click(fn=_jump_to_next_pending, outputs=outputs)
        batch_reject_btn.click(fn=_batch_reject_failed_quality, outputs=outputs)

        demo.load(fn=lambda: _load_sample(state_idx[0]), outputs=outputs)

    return demo


def run_stage3(config: Dict, manifest_path: str | Path) -> None:
    """Launch the Gradio review UI."""
    review_cfg = config.get("review", {})
    host: str = review_cfg.get("host", "127.0.0.1")
    port: int = review_cfg.get("port", 7860)
    share: bool = review_cfg.get("share", False)

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    log.info("Launching review UI at http://%s:%d  (share=%s)", host, port, share)

    app = build_app(manifest_path)
    app.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Launch TTS review UI")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.manifest)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
