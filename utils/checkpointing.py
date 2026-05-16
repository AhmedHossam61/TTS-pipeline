"""
utils/checkpointing.py
──────────────────────
SQLite-backed job state tracker for long-running TTS synthesis batches.

Each synthesis job (one prompt × one engine × one voice) is a row in the
`synthesis_jobs` table.  The pipeline can be killed and resumed at any time —
completed jobs are skipped automatically.
"""
from __future__ import annotations

import sqlite3
import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS synthesis_jobs (
    job_id      TEXT PRIMARY KEY,
    prompt_id   TEXT NOT NULL,
    engine      TEXT NOT NULL,
    voice       TEXT NOT NULL,
    text        TEXT NOT NULL,
    domain      TEXT,
    source      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    audio_path  TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_status ON synthesis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_engine ON synthesis_jobs(engine);
"""

JobStatus = str  # "pending" | "running" | "completed" | "failed"


class CheckpointDB:
    """Thread-safe SQLite checkpoint store for synthesis jobs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── private helpers ───────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(_SCHEMA)

    # ── public API ────────────────────────────────────────────────────────────

    def add_jobs(self, jobs: List[Dict]) -> int:
        """
        Insert new jobs.  Jobs whose job_id already exists are skipped
        (so calling add_jobs twice is safe / idempotent).

        Each dict must have keys: job_id, prompt_id, engine, voice, text.
        Optional keys: domain, source, audio_path.

        Returns the number of rows actually inserted.
        """
        inserted = 0
        now = self._now()
        with self._conn() as con:
            for j in jobs:
                try:
                    con.execute(
                        """
                        INSERT INTO synthesis_jobs
                            (job_id, prompt_id, engine, voice, text,
                             domain, source, audio_path, status, created_at, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,'pending',?,?)
                        """,
                        (
                            j["job_id"],
                            j["prompt_id"],
                            j["engine"],
                            j["voice"],
                            j["text"],
                            j.get("domain"),
                            j.get("source"),
                            j.get("audio_path"),
                            now,
                            now,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Already exists — skip silently
                    pass
        return inserted

    def get_pending(self, engine: Optional[str] = None, limit: int = 0) -> List[Dict]:
        """Return pending jobs, optionally filtered by engine."""
        with self._conn() as con:
            if engine:
                rows = con.execute(
                    "SELECT * FROM synthesis_jobs WHERE status='pending' AND engine=?"
                    + (f" LIMIT {limit}" if limit else ""),
                    (engine,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM synthesis_jobs WHERE status='pending'"
                    + (f" LIMIT {limit}" if limit else "")
                ).fetchall()
        return [dict(r) for r in rows]

    def get_failed(self, max_attempts: int = 3) -> List[Dict]:
        """Return failed jobs that haven't exceeded max_attempts."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM synthesis_jobs WHERE status='failed' AND attempts < ?",
                (max_attempts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_running(self, job_id: str) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE synthesis_jobs SET status='running', updated_at=? WHERE job_id=?",
                (self._now(), job_id),
            )

    def mark_completed(self, job_id: str, audio_path: str) -> None:
        with self._conn() as con:
            con.execute(
                """
                UPDATE synthesis_jobs
                SET status='completed', audio_path=?, updated_at=?
                WHERE job_id=?
                """,
                (audio_path, self._now(), job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._conn() as con:
            con.execute(
                """
                UPDATE synthesis_jobs
                SET status='failed', error=?,
                    attempts=attempts+1, updated_at=?
                WHERE job_id=?
                """,
                (error[:500], self._now(), job_id),
            )

    def reset_running(self) -> int:
        """
        Reset any jobs stuck in 'running' back to 'pending'.
        Call this on startup to recover from a previous crash.
        """
        with self._conn() as con:
            cur = con.execute(
                "UPDATE synthesis_jobs SET status='pending', updated_at=? WHERE status='running'",
                (self._now(),),
            )
        return cur.rowcount

    def get_stats(self) -> Dict[str, int]:
        """Return counts by status."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM synthesis_jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get_completed_audio_paths(self) -> Dict[str, str]:
        """Return {job_id: audio_path} for all completed jobs."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT job_id, audio_path FROM synthesis_jobs WHERE status='completed'"
            ).fetchall()
        return {r["job_id"]: r["audio_path"] for r in rows}

    def get_completed_jobs(self) -> Dict[str, Dict]:
        """Return {job_id: full_job_dict} for all completed jobs.

        The dict contains the original text, domain, source, engine, voice,
        and audio_path that were stored when the job was first registered and
        later completed.  Use this when building the manifest so the recorded
        text always matches what was actually synthesized.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT job_id, prompt_id, engine, voice, text, domain, source, audio_path "
                "FROM synthesis_jobs WHERE status='completed'"
            ).fetchall()
        return {r["job_id"]: dict(r) for r in rows}
