"""Tests for utils/checkpointing.py."""

from utils.checkpointing import CheckpointDB


def _sample_jobs():
    return [
        {
            "job_id": "job_1",
            "prompt_id": "p0001",
            "engine": "edge-tts",
            "voice": "ar-EG-SalmaNeural",
            "text": "إزيك النهارده؟",
            "domain": "conversation",
            "source": "seed",
            "audio_path": "data/audio/job_1.wav",
        },
        {
            "job_id": "job_2",
            "prompt_id": "p0002",
            "engine": "edge-tts",
            "voice": "ar-EG-ShakirNeural",
            "text": "عامل إيه؟",
            "domain": "conversation",
            "source": "seed",
            "audio_path": "data/audio/job_2.wav",
        },
        {
            "job_id": "job_3",
            "prompt_id": "p0003",
            "engine": "gemini-tts",
            "voice": "Charon",
            "text": "عايز أروح وسط البلد",
            "domain": "transport",
            "source": "seed",
            "audio_path": "data/audio/job_3.wav",
        },
    ]


def test_reset_running_recovers_stuck_jobs(tmp_path):
    """Simulate a mid-run crash: running jobs should return to pending."""
    db = CheckpointDB(tmp_path / "synthesis.db")
    jobs = _sample_jobs()
    db.add_jobs(jobs)

    # Simulate interrupted run: two jobs were marked running before crash.
    db.mark_running("job_1")
    db.mark_running("job_2")

    reset_count = db.reset_running()

    assert reset_count == 2

    pending_ids = {j["job_id"] for j in db.get_pending()}
    assert "job_1" in pending_ids
    assert "job_2" in pending_ids
    assert "job_3" in pending_ids


def test_reset_running_only_changes_running_status(tmp_path):
    """Ensure reset_running does not alter completed or failed jobs."""
    db = CheckpointDB(tmp_path / "synthesis.db")
    jobs = _sample_jobs()
    db.add_jobs(jobs)

    db.mark_running("job_1")
    db.mark_completed("job_2", "data/audio/job_2.wav")
    db.mark_failed("job_3", "synthetic failure")

    reset_count = db.reset_running()
    stats = db.get_stats()

    assert reset_count == 1
    assert stats.get("pending", 0) == 1
    assert stats.get("completed", 0) == 1
    assert stats.get("failed", 0) == 1