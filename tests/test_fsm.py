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
    """Stand-in for safety.VehicleTracker, honest to the three methods
    fsm.py actually calls on it: min_ttc(), unresolved_vehicles() and
    crossing_vehicles()."""

    def __init__(self, min_ttc=None, unresolved=0, crossing=None):
        self._min_ttc = min_ttc
        self._unresolved = unresolved
        # A plain list stand-in for the TrackState objects
        # safety.VehicleTracker.crossing_vehicles() returns -- fsm.py only
        # ever checks truthiness, so the contents don't matter here.
        self._crossing = crossing if crossing is not None else []

    def min_ttc(self):
        return self._min_ttc

    def unresolved_vehicles(self):
        return self._unresolved

    def crossing_vehicles(self):
        return self._crossing


CROSSWALK_CENTRE = make_detection("crosswalk", conf=0.9, x1=90.0, x2=110.0)  # centred in a 200px frame

# Post-review-fix taxonomy: ped_signal_* is what actually licenses SAFE;
# veh_signal_* and legacy signal_* (old models/best.pt weights, trained
# before the ped/veh split) are all WAIT-only evidence. SIGNAL_RED/
# SIGNAL_GREEN keep their old names as aliases to ped_signal_red/
# ped_signal_green so every pre-existing test below (which was written
# against the shared-class taxonomy and is still testing a real, unchanged
# rule -- e.g. "a red signal always wins") keeps meaning the same thing:
# "the pedestrian signal is red/green". Tests specifically about the new
# veh_signal_*/legacy split use the *_SIGNAL_GREEN/RED names below instead.
PED_SIGNAL_RED = make_detection("ped_signal_red", conf=0.9, x1=0.0, x2=10.0)
PED_SIGNAL_GREEN = make_detection("ped_signal_green", conf=0.9, x1=0.0, x2=10.0)
SIGNAL_RED = PED_SIGNAL_RED
SIGNAL_GREEN = PED_SIGNAL_GREEN
VEH_SIGNAL_RED = make_detection("veh_signal_red", conf=0.9, x1=0.0, x2=10.0)
VEH_SIGNAL_GREEN = make_detection("veh_signal_green", conf=0.9, x1=0.0, x2=10.0)
LEGACY_SIGNAL_GREEN = make_detection("signal_green", conf=0.9, x1=0.0, x2=10.0)  # old models/best.pt name
LEGACY_SIGNAL_RED = make_detection("signal_red", conf=0.9, x1=0.0, x2=10.0)  # old models/best.pt name
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
        def crossing_vehicles(self):
            return []

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


# ---------------------------------------------------------------------------
# Turning-vehicle rule (safety.py's crossing_vehicles()): forces WAITING
# with its own reason, and cannot be bypassed by the SAFE rule.
# ---------------------------------------------------------------------------

def test_crossing_vehicle_forces_waiting_even_with_green_and_clear_road():
    fsm = CrossingFSM()
    # A textbook-otherwise-safe road (no TTC, nothing unresolved, green
    # signal) except a vehicle flagged as crossing the path.
    tracker = FakeTracker(min_ttc=None, unresolved=0, crossing=["some_track"])
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, vehicle turning across the crossing"


def test_crossing_vehicle_beats_safe_regardless_of_detection_order():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0, crossing=["some_track"])
    detections = [SIGNAL_GREEN, CROSSWALK_CENTRE]
    verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, vehicle turning across the crossing"


def test_fsm_never_returns_safe_while_crossing_vehicle_present():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0, crossing=["some_track"])
    detections = [CROSSWALK_CENTRE, SIGNAL_GREEN]

    # Run well past the hysteresis window -- SAFE must never appear, in
    # either raw or committed form, the whole time a crossing vehicle is
    # present (same style as the red-light invariant test above).
    for _ in range(HYSTERESIS_FRAMES + 10):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state != SAFE_TO_CROSS
        assert verdict.state != SAFE_TO_CROSS
    assert verdict.state == WAITING
    assert verdict.reason == "wait, vehicle turning across the crossing"


def test_crossing_vehicle_does_not_block_searching_when_no_crosswalk():
    # Rule 1 (no crosswalk -> SEARCHING) still runs first -- a crossing
    # vehicle detected with no crosswalk in view has nothing to be "waiting
    # to cross" about yet.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0, crossing=["some_track"])
    verdict = fsm.update([SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == SEARCHING


def test_red_light_reason_still_wins_over_crossing_reason():
    # signal_red (rule 2) is checked before the crossing rule (rule 3), so
    # when both are true the red-light reason is reported -- this doesn't
    # change what fsm.py does (both are WAITING), just documents which
    # reason string surfaces, matching the fixed rule order in the module
    # docstring.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0, crossing=["some_track"])
    verdict = fsm.update([CROSSWALK_CENTRE, SIGNAL_RED], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, signal is red"


# ---------------------------------------------------------------------------
# Post-review-fix taxonomy split: ped_signal_* vs veh_signal_* vs legacy
# signal_* (old models/best.pt). This is the safety-critical part of the
# fix -- a green VEHICLE light must never be read as pedestrian permission
# to cross. See src/fsm.py's module docstring "POST-REVIEW FIX" section.
# ---------------------------------------------------------------------------

def test_veh_signal_green_forces_waiting_even_with_clear_road():
    # A textbook-otherwise-safe road (no TTC, nothing unresolved) but the
    # signal facing the CARS is green -- that is danger evidence for a
    # pedestrian (traffic has right of way), not permission, and must never
    # reach SAFE.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, VEH_SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, vehicle signal is green"
    assert verdict.raw_state != SAFE_TO_CROSS


def test_veh_signal_green_beats_safe_even_with_ped_signal_green_also_present():
    # If both a pedestrian green and a vehicle green show up in the same
    # frame (e.g. two signal heads in view), the conservative rule wins --
    # rule 2b is checked before rule 5 can ever look at ped_signal_green.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, PED_SIGNAL_GREEN, VEH_SIGNAL_GREEN], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, vehicle signal is green"


def test_kill_scenario_receding_vehicles_and_green_vehicle_light_is_waiting():
    # The exact kill scenario from the review: a vehicle light turns green,
    # cars ahead are driving AWAY from the camera so they track as
    # NOT_APPROACHING (min_ttc is None, receding vehicles don't produce a
    # TTC per safety.py), and there are no unresolved vehicles either. Under
    # the OLD shared-class rule this reached SAFE_TO_CROSS right as traffic
    # accelerated through the junction. It must now be WAITING.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)  # receding traffic: no TTC, nothing unresolved
    detections = [CROSSWALK_CENTRE, VEH_SIGNAL_GREEN]
    for _ in range(HYSTERESIS_FRAMES + 10):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state != SAFE_TO_CROSS
        assert verdict.state != SAFE_TO_CROSS
    assert verdict.state == WAITING
    assert verdict.reason == "wait, vehicle signal is green"


def test_ped_signal_green_and_clear_road_still_reaches_safe():
    # The system must not become incapable of ever saying SAFE: a genuine
    # pedestrian green, on a clear road, still reaches SAFE_TO_CROSS after
    # the hysteresis window, same as the old shared-class behaviour did for
    # a "real" (pedestrian) green.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    detections = [CROSSWALK_CENTRE, PED_SIGNAL_GREEN]
    verdict = None
    for _ in range(HYSTERESIS_FRAMES):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == SAFE_TO_CROSS
    assert verdict.state == SAFE_TO_CROSS
    assert verdict.reason == "safe to cross now"


def test_legacy_signal_green_from_old_weights_never_produces_safe():
    # models/best.pt predates the ped/veh split and still emits the bare old
    # name "signal_green". It must be treated conservatively as vehicle-
    # signal (WAIT) evidence, never as pedestrian permission -- this is the
    # single most important regression test in this file for a real,
    # currently-deployed model.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)  # otherwise-clear road
    detections = [CROSSWALK_CENTRE, LEGACY_SIGNAL_GREEN]
    for _ in range(HYSTERESIS_FRAMES + 10):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state != SAFE_TO_CROSS
        assert verdict.state != SAFE_TO_CROSS
    assert verdict.state == WAITING
    assert verdict.raw_reason == "wait, vehicle signal is green"


def test_legacy_signal_red_is_wait_evidence():
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    verdict = fsm.update([CROSSWALK_CENTRE, LEGACY_SIGNAL_RED], tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, signal is red"


def test_veh_signal_red_alone_does_not_produce_safe():
    # A stopped vehicle facing a red light says nothing about whether a
    # pedestrian has permission to cross -- no crosswalk-facing pedestrian
    # signal was ever seen green, so this must stay WAITING (via the
    # rule-6 default), never SAFE, however long it persists.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    detections = [CROSSWALK_CENTRE, VEH_SIGNAL_RED]
    for _ in range(HYSTERESIS_FRAMES + 10):
        verdict = fsm.update(detections, tracker, FRAME_WIDTH)
        assert verdict.raw_state != SAFE_TO_CROSS
        assert verdict.state != SAFE_TO_CROSS
    assert verdict.state == WAITING
    # Rule 2 fires ("wait, signal is red") since veh_signal_red is in
    # LEGACY_RED_WAIT_NAMES -- either way, the point being tested is that
    # this never becomes SAFE.
    assert verdict.reason == "wait, signal is red"


def test_veh_signal_red_does_not_unlock_safe_even_with_ped_signal_green():
    # Belt-and-braces: even if a ped_signal_green were also present in the
    # same frame as veh_signal_red, rule 2 (red beats everything) still
    # wins -- red is checked before any green, pedestrian or vehicle.
    fsm = CrossingFSM()
    tracker = FakeTracker(min_ttc=None, unresolved=0)
    detections = [CROSSWALK_CENTRE, VEH_SIGNAL_RED, PED_SIGNAL_GREEN]
    verdict = fsm.update(detections, tracker, FRAME_WIDTH)
    assert verdict.raw_state == WAITING
    assert verdict.raw_reason == "wait, signal is red"
