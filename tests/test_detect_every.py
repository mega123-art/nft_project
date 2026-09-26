"""
Tests for the --detect-every frame-skipping added to src/main.py to clear
PLAN.md's 8-12fps ground rule at imgsz 960 (measured 6.7-7.9fps at imgsz 960
without skipping -- see main.py's run() docstring).

Two things have to be proven, not just asserted in a comment:

1. (Phase 5) safety.py's TTC series must not change in any way that matters
   when the tracker is only fed on detection frames instead of every frame
   -- see test_ttc_agrees_between_detect_every_1_and_3 below, plus a
   companion test showing what would go wrong if a skipped frame *did*
   call tracker.update() with a stale, repeated box.

2. (Phase 6 + Phase 8 integration) src/main.py's run() must actually call
   the detector, tracker.update() and fsm.update() only on detection
   frames, carry the previous verdict/detections forward unchanged on
   skipped frames, and log every video frame with an accurate
   is_detection_frame flag -- see the FakeCap/FakeDetector-based tests at
   the bottom of this file, in the same synthetic/injection style as
   test_audio.py and test_decision_log.py.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety import VehicleTracker, MIN_SAMPLES_FOR_TTC
from detector import Detection
from fsm import SEARCHING, WAITING, SAFE_TO_CROSS, Verdict
from audio import AudioAnnouncer
import main as main_module


def make_detection(width, track_id=1, cls_name="car", centre_x=None, height=20.0):
    if centre_x is None:
        x1, x2 = 0.0, width
    else:
        x1, x2 = centre_x - width / 2.0, centre_x + width / 2.0
    return Detection(cls_name=cls_name, conf=0.9, x1=x1, y1=0.0, x2=x2, y2=height, track_id=track_id)


# ---------------------------------------------------------------------------
# Phase 5: TTC must be unaffected (beyond floating-point/window differences)
# by only feeding the tracker on a subset of frames.
# ---------------------------------------------------------------------------

D0 = 60.0
V = 10.0  # m/s closing speed -> contact at t = D0/V = 6.0s
K = 4000.0  # same choice as test_safety.py -- keeps the box comfortably
            # above the noise-horizon regime so this test is about TTC
            # accuracy, not the (separately tested) UNKNOWN-for-small-boxes
            # behaviour.
DT = 0.1


def _true_ttc(i):
    distance = D0 - V * (i * DT)
    return distance / V


def _width_at(i):
    distance = D0 - V * (i * DT)
    return K / distance


def test_ttc_agrees_between_detect_every_1_and_3():
    """This is the proof for main.py's run() choice to call
    tracker.update() ONLY on frames the detector actually ran on. Feed the
    same constant-velocity synthetic approach used in test_safety.py's TTC
    tests through two trackers: one fed every frame (detect_every=1) and
    one fed only every 3rd frame (detect_every=3 -- exactly the subset
    main.py's run() would feed a tracker at that setting; frame index i is
    a detection frame iff i % detect_every == 0, matching run()'s
    (frame_count - 1) % detect_every == 0 with frame_count = i + 1).

    Because this synthetic approach is perfectly constant-velocity (no
    noise), safety.py's reciprocal-width-vs-time straight-line fit
    recovers the true line exactly regardless of which subset of points it
    sees -- see safety.py's module docstring on why 1/width is exactly
    linear in time under constant velocity. So both series should match
    the analytic TTC almost exactly, not just "closely": this is a much
    tighter check than the general-purpose tolerance tests in
    test_safety.py, and specifically catches a regression that fed the
    tracker a stale, repeated box on skipped frames (see the companion
    test below for what that failure mode looks like).
    """
    n = 60
    detect_every = 3

    tracker_every1 = VehicleTracker()
    tracker_every3 = VehicleTracker()

    ttc_every1 = {}
    ttc_every3 = {}

    for i in range(n):
        t = i * DT
        tracker_every1.update([make_detection(_width_at(i), track_id=1)], t)
        ttc_every1[i] = tracker_every1.min_ttc()

        if i % detect_every == 0:
            tracker_every3.update([make_detection(_width_at(i), track_id=1)], t)
        # A caller (main.py) only ever reads min_ttc()/status on a
        # detection frame -- but sampling it here every i just documents
        # that the value simply doesn't change between detection frames,
        # which is the whole point of "not corrupting the tracker" on a
        # skip.
        ttc_every3[i] = tracker_every3.min_ttc()

    compared = 0
    for i in range(n):
        if i % detect_every != 0:
            continue
        a, b = ttc_every1[i], ttc_every3[i]
        if a is None or b is None:
            continue
        compared += 1
        assert abs(a - b) < 0.05, (
            f"TTC disagreement at i={i}: detect_every=1 -> {a:.4f}, "
            f"detect_every=3 -> {b:.4f}"
        )
        assert abs(a - _true_ttc(i)) < 0.05
        assert abs(b - _true_ttc(i)) < 0.05

    # Sanity: make sure this test actually exercised enough overlapping
    # readings to mean something, not just 0-1 lucky matches.
    assert compared >= 10


def test_feeding_tracker_on_skipped_frames_with_stale_box_corrupts_ttc():
    """The negative case: shows why main.py's run() must NOT call
    tracker.update() again on a skipped frame with the same (stale) box.

    This directly simulates the rejected design: every video frame calls
    tracker.update(), but on 2 out of every 3 frames the box handed in is
    the last REAL detection, unchanged, while the timestamp still advances
    (exactly what "detect every 3rd frame, keep updating the tracker every
    frame with whatever box we last saw" would do). Repeating a box while
    time moves on injects an artificial zero-growth sample -- du/dt = 0 for
    that pair -- which drags the fitted approach rate down and makes the
    reported TTC longer than the truth: optimistic, in exactly the
    direction PLAN.md forbids.
    """
    n = 60
    detect_every = 3

    tracker_correct = VehicleTracker()  # only fed on real detection frames
    tracker_wrong = VehicleTracker()    # fed every frame with a stale box

    last_width = None
    worst_wrong_error = 0.0
    worst_correct_error = 0.0

    for i in range(n):
        t = i * DT
        is_detection_frame = (i % detect_every == 0)
        if is_detection_frame:
            last_width = _width_at(i)
            tracker_correct.update([make_detection(last_width, track_id=1)], t)
        tracker_wrong.update([make_detection(last_width, track_id=1)], t)

        if not is_detection_frame:
            # Only compare at instants the "correct" tracker was actually
            # fed data. Its value is legitimately, boundedly stale between
            # detection frames (see run()'s docstring on the up-to
            # detect_every-1 frame lag from reusing detections on the
            # overlay) -- that lag is not the corruption this test is
            # isolating, so skip it here rather than conflate the two.
            continue

        true_ttc = _true_ttc(i)
        correct_ttc = tracker_correct.min_ttc()
        wrong_ttc = tracker_wrong.min_ttc()
        if correct_ttc is not None:
            worst_correct_error = max(worst_correct_error, abs(correct_ttc - true_ttc))
        if wrong_ttc is not None:
            worst_wrong_error = max(worst_wrong_error, abs(wrong_ttc - true_ttc))

    # The correct approach (skip = no update at all) stays essentially
    # exact; the wrong approach (skip = repeat the box) is measurably worse
    # -- proving the corruption is real, not hypothetical.
    assert worst_correct_error < 0.05
    assert worst_wrong_error > worst_correct_error + 0.2, (
        "expected repeating a stale box across skipped frames to visibly "
        "corrupt TTC accuracy relative to not updating the tracker at all"
    )


# ---------------------------------------------------------------------------
# Phase 6 + Phase 8 integration: main.py's run() itself.
# ---------------------------------------------------------------------------

class FakeCap:
    """Hands back a fixed number of blank frames, then reports EOF -- just
    enough of cv2.VideoCapture's surface for run() to drive its loop."""

    def __init__(self, n_frames, width=64, height=48):
        self._remaining = n_frames
        self._width = width
        self._height = height

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0


class FakeDetector:
    """Records every call and returns one crosswalk + one steadily
    approaching vehicle, so the FSM has something other than SEARCHING to
    chew on. Box width grows by a fixed step per call so
    tracker.update() sees genuine (if coarse) motion each time it's fed."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        width = 20.0 + 2.0 * self.calls
        return [
            Detection(cls_name="crosswalk", conf=0.9, x1=10.0, y1=10.0, x2=30.0, y2=20.0, track_id=None),
            Detection(cls_name="signal_green", conf=0.9, x1=0.0, y1=0.0, x2=5.0, y2=5.0, track_id=None),
            make_detection(width, track_id=1),
        ]


class RecordingLogger:
    def __init__(self):
        self.rows = []

    def log(self, frame_index, verdict, unresolved_vehicles, detections,
            inference_ms, total_ms, is_detection_frame=True):
        self.rows.append({
            "frame_index": frame_index,
            "verdict": verdict,
            "detections": detections,
            "inference_ms": inference_ms,
            "is_detection_frame": is_detection_frame,
        })


def test_detector_only_called_on_detection_frames():
    n_frames = 20
    detect_every = 4
    cap = FakeCap(n_frames)
    detector = FakeDetector()

    main_module.run(cap, headless=True, max_frames=None, detector=detector,
                     detect_every=detect_every)

    expected_calls = len(range(0, n_frames, detect_every))
    assert detector.calls == expected_calls


def test_skipped_frames_log_carried_forward_verdict_flagged_correctly():
    n_frames = 13
    detect_every = 3
    cap = FakeCap(n_frames)
    detector = FakeDetector()
    logger = RecordingLogger()

    main_module.run(cap, headless=True, max_frames=None, detector=detector,
                     decision_logger=logger, detect_every=detect_every)

    assert len(logger.rows) == n_frames

    last_real_verdict = None
    last_real_detections = None
    for row in logger.rows:
        i = row["frame_index"] - 1  # 0-based, matches run()'s own indexing
        expected_detection_frame = (i % detect_every == 0)
        assert row["is_detection_frame"] == expected_detection_frame

        if expected_detection_frame:
            assert row["inference_ms"] is not None
            last_real_verdict = row["verdict"]
            last_real_detections = row["detections"]
        else:
            # CRITICAL: a skipped frame must log the exact carried-forward
            # verdict/detections from the last real detection frame, not a
            # fresh (and impossible, since the detector didn't run)
            # re-evaluation, and not a null placeholder either -- see
            # main.py's run() docstring on why "carried forward" was
            # chosen over "null" or "silently nothing".
            assert row["inference_ms"] is None
            assert row["verdict"] is last_real_verdict
            assert row["detections"] is last_real_detections


def test_first_frame_is_always_a_detection_frame_regardless_of_detect_every():
    """The pipeline must not delay its very first look at the road just
    because detect_every > 1 -- frame_count is 1-based, so frame 1 must
    always satisfy (frame_count - 1) % detect_every == 0."""
    for detect_every in (1, 2, 5):
        cap = FakeCap(1)
        detector = FakeDetector()
        main_module.run(cap, headless=True, max_frames=None, detector=detector,
                         detect_every=detect_every)
        assert detector.calls == 1


def test_fsm_hysteresis_counts_detection_cycles_not_video_frames():
    """Phase 6's HYSTERESIS_FRAMES=8 is meant to require 8 consecutive real
    verdicts before entering SAFE_TO_CROSS. This checks main.py's run()
    keeps that meaning intact under frame skipping: with detect_every=4, a
    clip with fewer than 8*4=32 video frames must not be enough real
    detection cycles to ever reach SAFE_TO_CROSS, even though the FakeDetector
    consistently reports a clear, signal-green road (no vehicle blocking it
    -- see FakeDetector, which reports a receding/no-conflict vehicle isn't
    modelled here, so this test only needs enough frames that hysteresis,
    not evidence quality, is the limiting factor)."""
    detect_every = 4
    logger = RecordingLogger()

    # 27 video frames at detect_every=4 gives only 7 detection cycles
    # (i = 0, 4, 8, 12, 16, 20, 24), one short of HYSTERESIS_FRAMES=8 --
    # SAFE_TO_CROSS must not be reached yet if fsm.update() is correctly
    # gated to detection frames only (if it ran every video frame instead,
    # 27 frames would be far more than enough votes).
    cap = FakeCap(27)
    detector = FakeDetector()
    main_module.run(cap, headless=True, max_frames=None, detector=detector,
                     decision_logger=logger, detect_every=detect_every)

    states_seen = {row["verdict"].state for row in logger.rows}
    assert SAFE_TO_CROSS not in states_seen, (
        "SAFE_TO_CROSS reached on fewer than HYSTERESIS_FRAMES real detection "
        "cycles -- fsm.update() must only run on detection frames, not every "
        "video frame, or stale evidence gets recounted as fresh votes"
    )


# ---------------------------------------------------------------------------
# Phase 7: calling announcer.announce() on every video frame, even the
# skipped ones carrying the identical Verdict object forward, must not
# re-speak the same transition.
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self):
        self.said = []

    def say(self, message):
        self.said.append(message)

    def runAndWait(self):
        pass

    def stop(self):
        pass


def _wait_for(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_carried_forward_verdict_does_not_re_announce_every_skip_frame():
    """main.py's run() calls announcer.announce(verdict) on every video
    frame, including skip frames where verdict is the SAME Verdict object
    as the last detection frame (see run()'s docstring on this). That
    object's .changed flag stays True on every one of those repeated calls
    -- it is not re-derived -- so this test pins down what actually stops
    that from re-speaking the same announcement detect_every-1 extra times
    per transition: audio.py's own GLOBAL_COOLDOWN_S, not a fresh
    verdict.state comparison in main.py. If that constant or this call
    pattern ever changes, this is the test that would catch a returning
    "announces every skipped frame" regression.
    """
    engine = _FakeEngine()
    fake_now = {"t": 0.0}
    announcer = AudioAnnouncer(
        tts_engine_factory=lambda: engine,
        mixer_factory=lambda: None,  # exercise the speech path only
        now=lambda: fake_now["t"],
    )
    try:
        verdict = Verdict(
            state=WAITING, reason="wait, vehicle approaching",
            raw_state=WAITING, raw_reason="wait, vehicle approaching",
            min_ttc=3.0, crosswalk_offset=None, crosswalk_direction=None,
            changed=True,
        )
        # Simulate main.py's run() calling announce() with the identical
        # object on 3 consecutive skip frames, ~100ms apart (comfortably
        # inside GLOBAL_COOLDOWN_S=2.0s, matching the real pipeline's fps).
        for _ in range(4):
            announcer.announce(verdict)
            fake_now["t"] += 0.1

        assert _wait_for(lambda: len(engine.said) >= 1)
        time.sleep(0.1)  # let the worker thread drain fully before counting
        assert len(engine.said) == 1, (
            f"expected exactly one spoken announcement for one transition, "
            f"got {engine.said!r}"
        )
    finally:
        announcer.close()
