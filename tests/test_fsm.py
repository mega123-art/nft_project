"""
Phase 6 tests for src/fsm.py.

Synthetic, like tests/test_safety.py: we build Detection objects directly
(reusing detector.Detection, same as test_safety.py does) and a small fake
tracker that mirrors safety.VehicleTracker's real, used-by-fsm.py surface
(min_ttc() and unresolved_vehicles(), both no-arg methods) rather than
inventing a different shape. Each test encodes one specific rule or guard
from PLAN.md's Phase 6 section, not just "whatever the code currently does".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fsm import (
    CrossingFSM,
    SEARCHING,
    WAITING,
    SAFE_TO_CROSS,
    HYSTERESIS_FRAMES,
    ROAD_WIDTH_M,
    WALK_SPEED_MPS,
    CROSSING_MARGIN_S,
)
from safety import DECISION_TTC_THRESHOLD
from detector import Detection


DEFAULT_REQUIRED_CROSSING_TIME = ROAD_WIDTH_M / WALK_SPEED_MPS + CROSSING_MARGIN_S  # 13.0s


def make_detection(cls_name, conf=0.9, x1=40.0, x2=60.0, y1=0.0, y2=20.0):
    return Detection(cls_name=cls_name, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2, track_id=None)


class FakeTracker:
    """Stand-in for safety.VehicleTracker, honest to the only two methods
    fsm.py actually calls on it: min_ttc() and unresolved_vehicles()."""

    def __init__(self, min_ttc=None, unresolved=0):
        self._min_ttc = min_ttc
        self._unresolved = unresolved

    def min_ttc(self):
        return self._min_ttc

    def unresolved_vehicles(self):
        return self._unresolved


CROSSWALK_CENTRE = make_detection("crosswalk", conf=0.9, x1=90.0, x2=110.0)  # centred in a 200px frame
SIGNAL_RED = make_detection("signal_red", conf=0.9, x1=0.0, x2=10.0)
SIGNAL_GREEN = make_detection("signal_green", conf=0.9, x1=0.0, x2=10.0)
FRAME_WIDTH = 200


# ---------------------------------------------------------------------------
# The one invariant that matters most: red light never enters SAFE.
# ---------------------------------------------------------------------------

def test_red_light_never_enters_safe_even_with_green_and_clear_road():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)  # a textbook-clear road
    detections = [CROSSWALK_CENTRE, SIGNAL_RED, SIGNAL_GREEN]  # both signals "present"

    # Run well past the hysteresis window in both raw and committed terms --
    # PLAN.md says this must NEVER enter SAFE, not just "not immediately".
    # The committed `state` starts at SEARCHING and takes HYSTERESIS_FRAMES
    # to catch up to WAITING like any other non-SAFE transition -- what
    # must hold on EVERY single frame, with no debounce delay allowed, is
    # that neither raw_state nor state ever reads SAFE_TO_CROSS.
    for _ in range(HYSTERESIS_FRAMES + 10):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state == WAITING
        assert verdict.raw_reason == "wait, signal is red"
        assert verdict.state != SAFE_TO_CROSS
    # By now (well past the hysteresis window) the committed state caught up too.
    assert verdict.state == WAITING
    assert verdict.reason == "wait, signal is red"


def test_red_light_beats_green_regardless_of_order_in_detection_list():
    # Order in the detections list must not matter -- put green first.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    detections = [SIGNAL_GREEN, SIGNAL_RED, CROSSWALK_CENTRE]
    verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, signal is red"


# ---------------------------------------------------------------------------
# No crosswalk -> SEARCHING, regardless of everything else.
# ---------------------------------------------------------------------------

def test_no_crosswalk_means_searching_regardless_of_signals_or_ttc():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=1.0, unresolved=0)  # even a close vehicle
    detections = [SIGNAL_RED, SIGNAL_GREEN]  # both signals, no crosswalk at all
    verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == SEARCHING
    assert verdict.reason == "looking for a crossing"


def test_no_detections_at_all_means_searching_not_safe():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    verdict = fsm.update([], tracker, FRAME_WIDTH)
    assert verdict.raw_state == SEARCHING
    assert verdict.state == SEARCHING
    assert verdict.crosswalk_offset is None
    assert verdict.crosswalk_direction is None


# ---------------------------------------------------------------------------
# min_ttc vs the flat 5.0s threshold (rule 3), isolated from the separate,
# stricter crossing-time gate (rule 4) by using a tiny road_width_m so both
# thresholds coincide at exactly 5.0s for this test only.
# ---------------------------------------------------------------------------

def test_min_ttc_just_under_5s_is_waiting_vehicle_approaching():
    # road_width_m=2.0 -> required crossing time = 2.0/1.0 + 3.0 = 5.0s,
    # same as DECISION_TTC_THRESHOLD, so rule 3 alone decides the outcome
    # for ttc values on either side of 5.0s -- rule 4 would agree either way.
    fsm = CrossingFSM(road_width_m=2.0)
    tracker = FakeTracker(min_ttc=DECISION_TTC_THRESHOLD - 0.01, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, vehicle approaching"


def test_min_ttc_just_over_5s_is_not_the_approaching_waiting_reason():
    fsm = CrossingFSM(road_width_m=2.0)
    tracker = FakeTracker(min_ttc=DECISION_TTC_THRESHOLD + 0.01, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    # With the two thresholds coinciding at 5.0s in this test, crossing just
    # over 5.0s clears rule 3 AND rule 4 (green + road clear) -> SAFE (raw).
    assert verdict.raw_state == SAFE_TO_CROSS
    assert verdict.raw_reason != "wait, vehicle approaching"


# ---------------------------------------------------------------------------
# Crossing-time check: the rule most likely to be conflated with the 5.0s
# one. A gap of 6s clears rule 3 (>= 5.0s) but must NOT be SAFE at the
# default road width, because 6s < 10.0/1.0 + 3.0 = 13.0s.
# ---------------------------------------------------------------------------

def test_gap_over_5s_but_under_required_crossing_time_is_not_safe():
    assert DECISION_TTC_THRESHOLD < 6.0 < DEFAULT_REQUIRED_CROSSING_TIME  # sanity on the fixture itself
    fsm = CrossingFSM()  # default road_width_m=10.0 -> required 13.0s
    tracker = FakeTracker(min_ttc=6.0, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state != SAFE_TO_CROSS
    assert verdict.raw_state == WAITING


def test_gap_over_required_crossing_time_is_safe():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=DEFAULT_REQUIRED_CROSSING_TIME + 0.5, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == SAFE_TO_CROSS


# ---------------------------------------------------------------------------
# unresolved_vehicles() blocks SAFE even with green and no TTC at all.
# ---------------------------------------------------------------------------

def test_unresolved_vehicle_blocks_safe_even_with_green_and_no_ttc():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=1)  # a visible, unassessed vehicle
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state != SAFE_TO_CROSS
    assert verdict.raw_state == WAITING


# ---------------------------------------------------------------------------
# Hysteresis: 7 consecutive SAFE-raw verdicts do not commit; the 8th does.
# ---------------------------------------------------------------------------

def test_hysteresis_requires_eight_consecutive_frames_to_enter_safe():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    detections = [CROSSWALK_CENTRE, SIGNAL_GREEN]

    assert HYSTERESIS_FRAMES == 8  # this test is written against that exact number

    verdict = None
    for i in range(HYSTERESIS_FRAMES - 1):  # 7 frames
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state == SAFE_TO_CROSS  # raw agrees every time
        assert verdict.state == SEARCHING  # but committed state hasn't moved yet
        assert verdict.changed is False

    # 8th consecutive agreeing frame: now it commits.
    verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == SAFE_TO_CROSS
    assert verdict.state == SAFE_TO_CROSS
    assert verdict.changed is True


# ---------------------------------------------------------------------------
# Asymmetry: once SAFE, a single disagreeing (WAITING) frame leaves immediately.
# ---------------------------------------------------------------------------

def test_leaving_safe_is_immediate_not_vote_delayed():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    safe_detections = [CROSSWALK_CENTRE, SIGNAL_GREEN]

    for _ in range(HYSTERESIS_FRAMES):
        verdict = fsm.update(safe_detections, tracker, FRAME_WIDTH)
    assert verdict.state == SAFE_TO_CROSS  # confirm we actually got there first

    # A vehicle shows up: one single frame with signal_red now present.
    tracker_danger = FakeTracker(min_ttc=None, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_RED, SIGNAL_GREEN], tracker_danger, FRAME_WIDTH)
    assert verdict.state == WAITING
    assert verdict.changed is True
    assert verdict.reason == "wait, signal is red"


# ---------------------------------------------------------------------------
# Fail-safe: an exception from the tracker yields WAITING, not a crash, not SAFE.
# ---------------------------------------------------------------------------

def test_tracker_exception_yields_waiting_not_a_crash_not_safe():
    fsm = CrossingFSM()

    class ExplodingTracker:
        def min_ttc(self):
            raise RuntimeError("simulated tracker failure")

        def unresolved_vehicles(self):
            return 0

    # No signal_red present, so _decide() reaches the min_ttc() call that raises.
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], ExplodingTracker(), FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.state != SAFE_TO_CROSS
    assert verdict.raw_reason == "unclear, please wait (internal error)"


# ---------------------------------------------------------------------------
# Crosswalk offset sign and direction wording.
# ---------------------------------------------------------------------------

def test_crosswalk_offset_left():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    left_crosswalk = make_detection("crosswalk", conf=0.9, x1=0.0, x2=20.0)  # centre at x=10, frame centre=100
    verdict = fsm.update([left_crosswalk], tracker, FRAME_WIDTH)
    assert verdict.crosswalk_offset < 0
    assert verdict.crosswalk_direction == "left"


def test_crosswalk_offset_right():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    right_crosswalk = make_detection("crosswalk", conf=0.9, x1=180.0, x2=200.0)  # centre at x=190
    verdict = fsm.update([right_crosswalk], tracker, FRAME_WIDTH)
    assert verdict.crosswalk_offset > 0
    assert verdict.crosswalk_direction == "right"


def test_crosswalk_offset_centre():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    # centre at x=100, exactly frame centre for FRAME_WIDTH=200.
    verdict = fsm.update([CROSSWALK_CENTRE], tracker, FRAME_WIDTH)
    assert abs(verdict.crosswalk_offset) <= 0.1
    assert verdict.crosswalk_direction == "center"
