"""
Phase 8: per-frame decision logging to SQLite, for scripts/evaluate.py.

PLAN.md's own words for this file's job: "Log every frame's decision with
timestamp, state, confidence, min_ttc and the detected classes to
logs/decisions.db." This module is the write side of that; scripts/evaluate.py
is the read side.

WHY A BACKGROUND WRITER, NOT A DIRECT INSERT PER FRAME. A synchronous
`INSERT` + `commit()` on every frame calls fsync on the video's hot path --
measured on this machine, that alone cost noticeably more than the ~1 fps
budget PLAN.md allows for logging (see report/evaluation.md for the actual
before/after numbers). So DecisionLogger.log() never touches the database
itself: it only builds a small dict and puts it on a queue, then returns.
A single daemon thread owns the actual sqlite3.Connection and batches many
rows into one transaction, the same "never let the slow thing block the
loop" shape as audio.py's speech worker.

WHY A JSON COLUMN FOR DETECTIONS, NOT A SECOND TABLE. The number of
detections varies frame to frame, and every consumer of this log
(evaluate.py) wants "all detections this frame" as one unit, not joined back
together from a child table -- a proper `detections` table normalized on
`frame_id` would be the textbook-correct schema, but it turns one insert
into N+1 and buys nothing this project needs (we never query "all frames
containing class X" across millions of rows, we replay a few thousand rows
sequentially). A JSON array of `{cls_name, conf}` in one TEXT column is one
insert, self-describing, and trivially read back with `json.loads`. If a
query workload ever needs indexing into individual classes, `json_each` in
sqlite's json1 extension (bundled since 3.38) can query this column without
a schema change.

WHY A SCHEMA VERSION COLUMN, NOT A SEPARATE METADATA TABLE. It has to be
readable by evaluate.py without a second query, and it has to survive a
schema change made after some logs already exist on disk -- a plain
integer column stamped onto every row (SCHEMA_VERSION below) does both. If
Phase 8's follow-up ever needs to change the columns, evaluate.py can branch
on this value instead of guessing from what columns happen to exist.

IS_DETECTION_FRAME (added for main.py's --detect-every). When frame
skipping is enabled, most video frames carry the previous detection frame's
verdict and detections forward unchanged rather than re-running the
detector (see main.py's run() docstring for why). Those rows are still
logged -- dropping them would make compute_frame_rate()'s wall-clock fps in
scripts/evaluate.py undercount the real video frame rate, and ground-truth
frame_labels.csv is keyed by video frame index, not detection cycle -- but
they are logged with is_detection_frame=0 so evaluate.py can tell "fresh
evidence" from "carried forward" apart: the latter must not be double
counted as new evidence when checking, say, how quickly the system reacts.

GRACEFUL DEGRADATION, matching audio.py's shape exactly: any failure to
open the database, create the writer thread, or write a batch is logged
once as a warning and turns this logger into a no-op for the rest of its
life. A full disk, a locked file, a bad path -- none of it may be allowed
to take down the capture loop, because PLAN.md's ground rule is "the vision
system must keep running," and a debug log is not allowed to violate that.
"""

import json
import logging
import os
import queue
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

# Bump this and extend CREATE TABLE below if the columns ever change; keep
# old rows readable by branching on the value evaluate.py sees, rather than
# guessing from which columns happen to exist in an old logs/decisions.db.
# Bumped to 2 for is_detection_frame (see main.py's --detect-every): a row
# with is_detection_frame == 0 carries the *previous* detection frame's
# verdict/detections forward unchanged (see main.py's run() docstring), not
# a fresh evaluation of this video frame -- a consumer that averages
# raw_state agreement or inference latency across rows needs to know that,
# rather than silently double-counting stale evidence as new.
SCHEMA_VERSION = 2

DEFAULT_DB_PATH = "logs/decisions.db"

# How many queued rows to write per transaction, and the longest we'll let
# rows sit unwritten when frames are arriving slower than this. Both exist
# for the same reason: one commit per frame is what makes logging slow (see
# module docstring); batching amortizes the fsync cost across many rows.
BATCH_SIZE = 50
FLUSH_INTERVAL_S = 0.5

# Rows queued but not yet written by the background thread. Bounded so a
# stuck/dead writer thread can't grow this without limit -- if the queue is
# ever full we drop the newest row rather than block the capture loop (see
# _enqueue below), the same trade audio.py makes on its speech queue.
MAX_QUEUE_SIZE = 2000

_SENTINEL = object()  # pushed to ask the writer thread to flush and exit

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    frame_index INTEGER NOT NULL,
    video_source TEXT,
    state TEXT NOT NULL,
    raw_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    min_ttc REAL,
    unresolved_vehicles INTEGER,
    crosswalk_offset REAL,
    crosswalk_direction TEXT,
    detections_json TEXT NOT NULL,
    inference_ms REAL,
    total_ms REAL,
    is_detection_frame INTEGER NOT NULL DEFAULT 1
)
"""


def _row_from_verdict(frame_index, video_source, verdict, unresolved_vehicles,
                       detections, inference_ms, total_ms, is_detection_frame):
    """Build the plain dict that gets queued. Kept as one small function so
    DecisionLogger.log() and the tests agree on exactly what a row contains."""
    detections_payload = [
        {"cls_name": d.cls_name, "conf": d.conf} for d in detections
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "wall_time": time.time(),
        "frame_index": frame_index,
        "video_source": video_source,
        "state": verdict.state,
        "raw_state": verdict.raw_state,
        "reason": verdict.reason,
        "min_ttc": verdict.min_ttc,
        "unresolved_vehicles": unresolved_vehicles,
        "crosswalk_offset": verdict.crosswalk_offset,
        "crosswalk_direction": verdict.crosswalk_direction,
        "detections_json": json.dumps(detections_payload),
        "inference_ms": inference_ms,
        "total_ms": total_ms,
        "is_detection_frame": 1 if is_detection_frame else 0,
    }


class DecisionLogger:
    """Queues one row per frame and writes it to SQLite on a background
    thread, in batches. Never blocks the caller beyond a queue put, and
    never raises out of log()/close() -- see module docstring.

    connection_factory is injectable so tests can point this at a
    throwaway file (or exercise failure paths) without touching real
    disk state -- the same pattern audio.py uses for tts_engine_factory /
    mixer_factory.
    """

    def __init__(self, db_path=DEFAULT_DB_PATH, video_source=None,
                 connection_factory=None, batch_size=BATCH_SIZE,
                 flush_interval=FLUSH_INTERVAL_S):
        self.video_source = video_source
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._dropped_rows = 0  # count only, so a flood of drops logs once, not per-row
        self._warned_about_drops = False

        self._enabled = False
        self._thread = None

        try:
            if connection_factory is not None:
                conn = connection_factory()
            else:
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
                conn = sqlite3.connect(db_path, check_same_thread=False)
                # WAL + NORMAL synchronous trade a small durability window
                # (a hard crash could lose the last fraction-of-a-second of
                # rows) for a lot less per-commit fsync cost -- an
                # acceptable trade for a debug/evaluation log, not for
                # anything the FSM itself depends on to be safe.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        except Exception:
            logger.warning(
                "DecisionLogger: failed to open %s, logging disabled", db_path,
                exc_info=True,
            )
            return

        self._conn = conn
        self._enabled = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    @property
    def enabled(self):
        return self._enabled

    def log(self, frame_index, verdict, unresolved_vehicles, detections,
            inference_ms, total_ms, is_detection_frame=True):
        """Queue one frame's row. Never raises, never blocks meaningfully.

        is_detection_frame is True for a frame the detector actually ran
        on, and False for a --detect-every skip frame where verdict and
        detections are the previous detection frame's values carried
        forward unchanged (see main.py's run() docstring). Defaulting to
        True keeps every existing caller (detect-every disabled, and every
        pre-existing test) logging exactly as before.
        """
        if not self._enabled:
            return
        try:
            row = _row_from_verdict(
                frame_index, self.video_source, verdict, unresolved_vehicles,
                detections, inference_ms, total_ms, is_detection_frame,
            )
            self._enqueue(row)
        except Exception:
            logger.warning("DecisionLogger: log() failed, dropping this row", exc_info=True)

    def _enqueue(self, row):
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # The writer thread is falling behind (or dead). Drop the
            # newest row rather than block the capture loop -- a gap in a
            # debug log is far cheaper than a stutter in the video the log
            # is trying to describe. Warn once, not once per dropped row,
            # so a sustained backlog doesn't itself become a logging flood.
            self._dropped_rows += 1
            if not self._warned_about_drops:
                logger.warning(
                    "DecisionLogger: queue full, dropping rows (writer thread "
                    "falling behind)"
                )
                self._warned_about_drops = True

    def _writer_loop(self):
        """Background thread: pull rows off the queue, INSERT them in
        batches. One transaction per batch is the whole point -- see module
        docstring on why a commit-per-frame is too slow."""
        batch = []
        last_flush = time.monotonic()
        while True:
            timeout = max(0.0, self._flush_interval - (time.monotonic() - last_flush))
            try:
                item = self._queue.get(timeout=timeout if timeout > 0 else None)
            except queue.Empty:
                item = None

            stop = False
            if item is _SENTINEL:
                stop = True
            elif item is not None:
                batch.append(item)

            due_by_size = len(batch) >= self._batch_size
            due_by_time = batch and (time.monotonic() - last_flush) >= self._flush_interval
            if batch and (due_by_size or due_by_time or stop):
                self._write_batch(batch)
                batch = []
                last_flush = time.monotonic()

            if stop:
                return

    def _write_batch(self, batch):
        try:
            self._conn.executemany(
                """
                INSERT INTO decisions (
                    schema_version, wall_time, frame_index, video_source,
                    state, raw_state, reason, min_ttc, unresolved_vehicles,
                    crosswalk_offset, crosswalk_direction, detections_json,
                    inference_ms, total_ms, is_detection_frame
                ) VALUES (
                    :schema_version, :wall_time, :frame_index, :video_source,
                    :state, :raw_state, :reason, :min_ttc, :unresolved_vehicles,
                    :crosswalk_offset, :crosswalk_direction, :detections_json,
                    :inference_ms, :total_ms, :is_detection_frame
                )
                """,
                batch,
            )
            self._conn.commit()
        except Exception:
            # A write failure must not kill the background thread -- the
            # next batch should still get a chance. The rows in this batch
            # are lost; that is the trade this whole module makes (see
            # module docstring: logging must never be allowed to matter
            # more than the capture loop staying up).
            logger.warning("DecisionLogger: batch write failed, dropping %d rows",
                            len(batch), exc_info=True)

    def close(self, timeout=2.0):
        """Flush pending rows and stop the writer thread. Safe to call more
        than once, and safe to call even if __init__ failed."""
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                # Make room, then push the sentinel -- it must get through
                # or the writer thread never exits.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
            self._thread.join(timeout=timeout)
        try:
            self._conn.close()
        except Exception:
            pass
        self._enabled = False
