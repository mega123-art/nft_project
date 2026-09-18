"""
Phase 8 tests for src/decision_log.py.

Like test_audio.py, these are synthetic and injection-based: each test opens
a throwaway sqlite file (pytest's tmp_path) rather than touching
logs/decisions.db, and the "failure degrades gracefully" tests use a
connection_factory that raises on purpose. Each test encodes one specific
promise DecisionLogger makes (schema created, rows written, batching
happens, a failure never reaches the caller) rather than "whatever the code
currently does".
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decision_log import DecisionLogger, SCHEMA_VERSION
from fsm import Verdict, SEARCHING, WAITING, SAFE_TO_CROSS
from detector import Detection


def make_verdict(state=WAITING, reason="wait, vehicle approaching", min_ttc=None,
                  offset=None, direction=None, raw_state=None, raw_reason=None):
    return Verdict(
        state=state,
        reason=reason,
        raw_state=raw_state if raw_state is not None else state,
        raw_reason=raw_reason if raw_reason is not None else reason,
        min_ttc=min_ttc,
        crosswalk_offset=offset,
        crosswalk_direction=direction,
        changed=True,
    )


def make_detection(cls_name="car", conf=0.8):
    return Detection(cls_name=cls_name, conf=conf, x1=0.0, y1=0.0, x2=10.0, y2=10.0, track_id=1)


def wait_for_rows(db_path, expected_count, timeout=2.0):
    """Poll the DB until expected_count rows land or timeout -- the writer
    thread is asynchronous by design, so tests must wait for it rather than
    assume a fixed sleep is long enough."""
    deadline = time.monotonic() + timeout
    last_count = -1
    while time.monotonic() < deadline:
        try:
            conn = sqlite3.connect(db_path)
            last_count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            conn.close()
        except sqlite3.OperationalError:
            last_count = -1
        if last_count >= expected_count:
            return last_count
        time.sleep(0.02)
    return last_count


# ---------------------------------------------------------------------------
# Schema + basic writing
# ---------------------------------------------------------------------------

def test_schema_created_on_open(tmp_path):
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source="test.mp4")
    try:
        assert logger.enabled
        conn = sqlite3.connect(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        conn.close()
        expected = {
            "id", "schema_version", "wall_time", "frame_index", "video_source",
            "state", "raw_state", "reason", "min_ttc", "unresolved_vehicles",
            "crosswalk_offset", "crosswalk_direction", "detections_json",
            "inference_ms", "total_ms",
        }
        assert expected <= cols
    finally:
        logger.close()


def test_rows_written_with_expected_values(tmp_path):
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4", batch_size=1)
    verdict = make_verdict(
        state=SAFE_TO_CROSS, reason="safe to cross now", min_ttc=6.5,
        offset=0.3, direction="right",
    )
    detections = [make_detection("car", 0.91), make_detection("crosswalk", 0.77)]
    logger.log(42, verdict, unresolved_vehicles=0, detections=detections,
               inference_ms=12.5, total_ms=15.0)
    try:
        assert wait_for_rows(db_path, 1) == 1
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM decisions").fetchone()
        conn.close()

        assert row["schema_version"] == SCHEMA_VERSION
        assert row["frame_index"] == 42
        assert row["video_source"] == "clip.mp4"
        assert row["state"] == SAFE_TO_CROSS
        assert row["reason"] == "safe to cross now"
        assert row["min_ttc"] == pytest.approx(6.5)
        assert row["unresolved_vehicles"] == 0
        assert row["crosswalk_offset"] == pytest.approx(0.3)
        assert row["crosswalk_direction"] == "right"
        assert row["inference_ms"] == pytest.approx(12.5)
        assert row["total_ms"] == pytest.approx(15.0)

        classes = json.loads(row["detections_json"])
        assert {c["cls_name"] for c in classes} == {"car", "crosswalk"}
    finally:
        logger.close()


def test_multiple_rows_batched_and_all_land(tmp_path):
    db_path = str(tmp_path / "decisions.db")
    # A small batch size forces at least one mid-run flush, not just the
    # final flush on close() -- this is what actually exercises batching
    # rather than merely "close() flushes everything."
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4", batch_size=3)
    for i in range(10):
        logger.log(i, make_verdict(state=SEARCHING, reason="looking for a crossing"),
                   unresolved_vehicles=0, detections=[], inference_ms=1.0, total_ms=2.0)
    try:
        assert wait_for_rows(db_path, 10) == 10
    finally:
        logger.close()


def test_close_flushes_pending_rows(tmp_path):
    db_path = str(tmp_path / "decisions.db")
    # Large batch size + long flush interval so nothing would be written
    # before close() is called, other than by close()'s own flush.
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4",
                             batch_size=1000, flush_interval=10.0)
    for i in range(5):
        logger.log(i, make_verdict(), unresolved_vehicles=0, detections=[],
                   inference_ms=1.0, total_ms=2.0)
    logger.close()

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    conn.close()
    assert count == 5


def test_close_is_safe_to_call_twice(tmp_path):
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4")
    logger.close()
    logger.close()  # must not raise


# ---------------------------------------------------------------------------
# Graceful degradation -- must never crash the caller
# ---------------------------------------------------------------------------

def test_construction_failure_disables_logger_without_raising():
    def bad_factory():
        raise RuntimeError("disk is on fire")

    logger = DecisionLogger(connection_factory=bad_factory)
    assert not logger.enabled

    # log() and close() on a disabled logger must be silent no-ops.
    logger.log(1, make_verdict(), unresolved_vehicles=0, detections=[],
               inference_ms=1.0, total_ms=2.0)
    logger.close()


def test_log_after_connection_closed_does_not_raise(tmp_path):
    """Simulates a mid-life failure (e.g. the writer thread's connection
    dies): the background write fails, but log() and close() from the
    caller's side must never raise."""
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4", batch_size=1)
    # Sabotage the connection the writer thread uses, from the test thread.
    logger._conn.close()

    logger.log(1, make_verdict(), unresolved_vehicles=0, detections=[],
               inference_ms=1.0, total_ms=2.0)
    time.sleep(0.2)  # give the writer thread a chance to hit the failure
    logger.close()  # must not raise even though self._conn.close() again fails internally


def test_log_never_raises_on_malformed_detection(tmp_path):
    """A detection missing the attributes DecisionLogger expects must
    degrade to a dropped row, not a crash of the capture loop."""
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4")
    try:
        logger.log(1, make_verdict(), unresolved_vehicles=0,
                   detections=[object()], inference_ms=1.0, total_ms=2.0)  # no .cls_name/.conf
    finally:
        logger.close()


def test_no_log_db_created_when_queue_full_is_handled(tmp_path):
    """Filling the queue beyond MAX_QUEUE_SIZE must drop rows, not block or
    raise. We use a tiny custom queue size via monkeypatching batch/flush
    to keep the writer thread from draining it, then flood it."""
    db_path = str(tmp_path / "decisions.db")
    # Enormous flush_interval and batch_size so the writer thread won't
    # drain anything before we've queued a burst -- exercises the "queue
    # full -> drop, warn once" path without waiting on real disk speed.
    logger = DecisionLogger(db_path=db_path, video_source="clip.mp4",
                             batch_size=1_000_000, flush_interval=1000.0)
    import decision_log as decision_log_module
    original_max = decision_log_module.MAX_QUEUE_SIZE
    logger._queue.maxsize = 3
    try:
        for i in range(20):
            logger.log(i, make_verdict(), unresolved_vehicles=0, detections=[],
                       inference_ms=1.0, total_ms=2.0)  # must never raise/block
        assert logger._dropped_rows > 0
    finally:
        decision_log_module.MAX_QUEUE_SIZE = original_max
        logger.close()
