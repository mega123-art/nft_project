"""
Phase 6: turn detections + Phase 5's tracker into one crossing verdict.

This module owns no vision and no tracking of its own -- it only reads what
src/detector.py and src/safety.py already produced for this frame and
applies PLAN.md's decision rules, in order, plus the extra guards PLAN.md
calls out by name: hysteresis, a crossing-time check, and a fail-safe
wrapper. Plain Python, no state-machine library -- every branch here has to
be explainable line by line in the viva.

THE GROUND RULE THIS WHOLE FILE SERVES (from PLAN.md): the default output is
always "wait". Only explicit positive evidence produces "safe to cross". A
false SAFE is the only unacceptable failure mode, so every ambiguous case in
this file resolves to WAITING, never to SAFE_TO_CROSS.

STATES. PLAN.md names six: SEARCHING -> AT_CROSSING -> WAITING ->
SAFE_TO_CROSS -> CROSSING -> DONE. The five decision rules PLAN.md actually
specifies only ever emit three of them: SEARCHING, WAITING, SAFE_TO_CROSS.
AT_CROSSING, CROSSING and DONE describe a lifecycle this module cannot
observe from a single frame of detections: AT_CROSSING would mean "we found
a crosswalk and are still evaluating it", which the rule table folds
straight into WAITING/SAFE_TO_CROSS rather than treating as a distinct
output; CROSSING (the user is now mid-crossing) and DONE (they've reached
the other side) both require knowing the user's own position and progress,
which a single fixed camera watching the road has no way to observe -- that
needs a Phase 7+ input (e.g. the user confirming "starting to cross" and
"made it"), not a vision signal. The three constants are still defined
below so main.py, tests, and later phases have one shared vocabulary, but
update() only ever returns SEARCHING, WAITING or SAFE_TO_CROSS. This is
flagged again in the project report rather than silently guessed at.

FAIL-SAFE INPUTS PLAN.md NAMES THAT DON'T EXIST YET: "low-confidence frame"
and "camera-tilt rejection" are listed in PLAN.md as fail-safe triggers, but
no upstream module in this repo currently produces either signal -- there is
no frame-level confidence score and no tilt detector anywhere in
detector.py/safety.py. Rather than invent a fake signal, this module uses
the one piece of real confidence information it has (each Detection's own
`conf`, from detector.py) as the practical stand-in: crosswalk and
signal_green -- the two classes whose presence pushes toward SAFE -- are
only trusted above POSITIVE_EVIDENCE_MIN_CONF; signal_red is trusted at any
confidence the detector already passed, because a false-positive red only
ever produces extra, harmless caution, never a false SAFE. The try/except in
update() is the hook a future camera-tilt signal would raise into.
"""

from dataclasses import dataclass
from typing import Optional

from safety import DECISION_TTC_THRESHOLD

# --- States -----------------------------------------------------------
# See the module docstring: only SEARCHING, WAITING and SAFE_TO_CROSS are
# ever returned by update() today. AT_CROSSING/CROSSING/DONE are reserved
# for later phases that have a way to observe the user's own progress.
SEARCHING = "SEARCHING"
AT_CROSSING = "AT_CROSSING"
WAITING = "WAITING"
SAFE_TO_CROSS = "SAFE_TO_CROSS"
CROSSING = "CROSSING"
DONE = "DONE"

# --- Hysteresis ---------------------------------------------------------
# 8 consecutive frames at ~25-30fps camera / 8-15fps inference is roughly
# 0.3s at this pipeline's measured rate (see PLAN.md Phase 6) -- long enough
# to ignore a single flickered detection, short enough not to feel laggy.
HYSTERESIS_FRAMES = 8

# --- Crossing-time check --------------------------------------------------
# Placeholder config constant -- PLAN.md explicitly allows this for now.
# 10.0m is a rough two-lane-plus-shoulder Indian road width. This MUST be
# measured per physical crossing once there is a fixed camera installation
# to measure it at; treat this number as a stand-in, not calibration.
ROAD_WIDTH_M = 10.0

# Deliberately slow and conservative -- an average adult walks faster than
# this. Slower assumed speed -> longer required gap -> harder to reach SAFE.
WALK_SPEED_MPS = 1.0

# Covers reaction time before stepping off, plus the fact that TTC on an
# accelerating vehicle is itself optimistic (see safety.py's module
# docstring) -- this margin is the plain, explainable way to eat into that
# optimism rather than pretending safety.py's number is exact.
CROSSING_MARGIN_S = 3.0

# --- Confidence gate for positive evidence -------------------------------
# See the module docstring's note on the "low-confidence frame" fail-safe:
# this is the practical stand-in, applied only to the two classes whose
# presence pushes toward SAFE (crosswalk, signal_green). signal_red is
# deliberately NOT gated by this -- see _decide().
POSITIVE_EVIDENCE_MIN_CONF = 0.6

# A crosswalk detection within this fraction of frame-half-width either side
# of centre reads as "centre" rather than left/right -- without a dead band
# a crosswalk sitting exactly on the centreline would flicker between
# "slightly left" and "slightly right" on pixel noise alone.
OFFSET_CENTRE_BAND = 0.1


@dataclass
class Verdict:
    """Everything one frame's decision produces.

    state / reason are the HYSTERESIS-STABILISED verdict actually in
    effect this frame -- this is what Phase 7 should look at, and it should
    only speak when `changed` is True (see CrossingFSM._apply_hysteresis).
    raw_state / raw_reason are the un-debounced, this-frame-only rule
    evaluation, kept around for Phase 8's per-frame logging and for tests
    that want to check rule boundaries without waiting out the hysteresis
    window.
    min_ttc / crosswalk_offset / crosswalk_direction are always this
    frame's real values regardless of hysteresis, again for Phase 8.
    """

    state: str
    reason: str
    raw_state: str
    raw_reason: str
    min_ttc: Optional[float]
    crosswalk_offset: Optional[float]  # signed, -1 (left edge) .. +1 (right edge)
    crosswalk_direction: Optional[str]  # "left" / "center" / "right"
    changed: bool  # True only on the frame `state` actually changed


def _best_detection(detections, cls_name, min_conf=0.0):
    """Highest-confidence detection of cls_name with conf >= min_conf, or None."""
    candidates = [d for d in detections if d.cls_name == cls_name and d.conf >= min_conf]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.conf)


def _crosswalk_offset(crosswalk_det, frame_width):
    """Signed horizontal offset of a crosswalk box from frame centre, plus
    a human-readable direction. None, None if there is nothing to report."""
    if crosswalk_det is None or not frame_width:
        return None, None

    centre_x = (crosswalk_det.x1 + crosswalk_det.x2) / 2.0
    half_width = frame_width / 2.0
    offset = (centre_x - half_width) / half_width
    # Clamp -- a box that straddles the frame edge could otherwise produce
    # a number outside [-1, 1], which is meaningless as "how far off centre".
    offset = max(-1.0, min(1.0, offset))

    if offset < -OFFSET_CENTRE_BAND:
        direction = "left"
    elif offset > OFFSET_CENTRE_BAND:
        direction = "right"
    else:
        direction = "center"
    return offset, direction


class CrossingFSM:
    """Turns one frame's (detections, tracker) into a debounced Verdict.

    Call update() once per frame with the same VehicleTracker instance that
    has already been fed this frame's detections (main.py does
    `tracker.update(detections, t)` before calling this) -- this module
    reads the tracker, it does not feed it.
    """

    def __init__(self, road_width_m=ROAD_WIDTH_M):
        self.road_width_m = road_width_m

        # Committed (hysteresis-stabilised) state -- what update() returns
        # as `state` until enough consecutive frames agree on something
        # else. Starts at SEARCHING: we have seen nothing yet, and
        # SEARCHING is the state PLAN.md's own rules produce for "nothing
        # found" -- never start pre-loaded into anything that could look
        # like positive evidence.
        self._state = SEARCHING
        self._reason = "looking for a crossing"

        # Vote buffer for hysteresis: the candidate state waiting to become
        # committed, and how many consecutive frames have agreed on it.
        self._vote_state = None
        self._vote_count = 0

    def update(self, detections, tracker, frame_width):
        """Return this frame's Verdict. Never raises."""
        try:
            raw_state, raw_reason, min_ttc, offset, direction = self._decide(
                detections, tracker, frame_width
            )
        except Exception:
            # FAIL SAFE: any exception anywhere in decision-making -- a
            # malformed detection, a tracker call blowing up, anything --
            # must not crash the capture loop and must never be read as
            # SAFE by default. This except is deliberately bare: we do not
            # get to pick and choose which exceptions are "safe" to ignore.
            raw_state = WAITING
            raw_reason = "unclear, please wait (internal error)"
            min_ttc = None
            offset = None
            direction = None

        changed = self._apply_hysteresis(raw_state, raw_reason)

        return Verdict(
            state=self._state,
            reason=self._reason,
            raw_state=raw_state,
            raw_reason=raw_reason,
            min_ttc=min_ttc,
            crosswalk_offset=offset,
            crosswalk_direction=direction,
            changed=changed,
        )

    def _decide(self, detections, tracker, frame_width):
        """PLAN.md's five rules, in order. Returns
        (raw_state, raw_reason, min_ttc, crosswalk_offset, crosswalk_direction)
        for THIS frame only -- no hysteresis applied here."""
        crosswalk_det = _best_detection(detections, "crosswalk", POSITIVE_EVIDENCE_MIN_CONF)
        offset, direction = _crosswalk_offset(crosswalk_det, frame_width)

        # Rule 1: no crosswalk detected -> SEARCHING. Checked first and
        # returns immediately -- with no crossing in view there is nothing
        # else worth evaluating, and "nothing found" must never fall
        # through toward SAFE by accident.
        if crosswalk_det is None:
            return SEARCHING, "looking for a crossing", None, offset, direction

        # Rule 2: signal_red present -> WAITING, unconditionally. No
        # confidence gate here on purpose (see module docstring): a
        # false-positive red only ever produces extra caution, so there is
        # no safety reason to demand a higher bar for it, and demanding one
        # would risk missing a real red. This check happens BEFORE min_ttc
        # or signal_green are even looked at, so nothing below can override
        # a red light -- this is what makes the red-light test unmissable.
        if any(d.cls_name == "signal_red" for d in detections):
            return WAITING, "wait, signal is red", tracker.min_ttc(), offset, direction

        min_ttc = tracker.min_ttc()

        # Rule 3: a vehicle is close enough that PLAN.md's flat 5.0s
        # threshold alone says wait, regardless of anything else.
        if min_ttc is not None and min_ttc < DECISION_TTC_THRESHOLD:
            return WAITING, "wait, vehicle approaching", min_ttc, offset, direction

        # Rule 4: green signal AND a road we can honestly call clear.
        signal_green = _best_detection(detections, "signal_green", POSITIVE_EVIDENCE_MIN_CONF)
        if signal_green is not None and self._road_is_clear(min_ttc, tracker.unresolved_vehicles()):
            return SAFE_TO_CROSS, "safe to cross now", min_ttc, offset, direction

        # Rule 5: otherwise -- e.g. no signal detected at all, signal_green
        # too low-confidence to trust, or the road fails the crossing-time
        # or unresolved-vehicle check. PLAN.md's default is always "wait".
        return WAITING, "unclear, please wait", min_ttc, offset, direction

    def _road_is_clear(self, min_ttc, unresolved_vehicles):
        """A road counts as clear only when ALL of the following hold.

        1. unresolved_vehicles == 0 -- no vehicle currently visible in frame
           that safety.py cannot yet assess (its UNKNOWN status). A vehicle
           we cannot assess is not evidence of anything, in either
           direction -- see VehicleTracker.unresolved_vehicles()'s own
           caller contract in safety.py's docstring: Phase 6 must never
           declare SAFE while this is non-zero.
        2. min_ttc is None (no vehicle is currently assessed as
           APPROACHING at all -- receding/parked vehicles do not count,
           per safety.py), OR min_ttc is comfortably above the time this
           crossing actually takes to walk (see _required_crossing_time),
           not merely above PLAN.md's 5.0s alarm threshold. 5.0s is the
           point at which an already-close vehicle demands an immediate
           WAITING; it says nothing about whether there is enough time to
           WALK the whole crossing, which typically takes much longer.
        """
        if unresolved_vehicles != 0:
            return False
        if min_ttc is None:
            return True
        return min_ttc > self._required_crossing_time()

    def _required_crossing_time(self):
        """Estimated seconds to clear the crossing, plus a safety margin.

        road_width_m is a placeholder (see ROAD_WIDTH_M) pending a measured
        value per physical crossing. WALK_SPEED_MPS is deliberately slow.
        CROSSING_MARGIN_S covers reaction time and safety.py's documented
        optimism on an accelerating vehicle's TTC.
        """
        return self.road_width_m / WALK_SPEED_MPS + CROSSING_MARGIN_S

    def _apply_hysteresis(self, raw_state, raw_reason):
        """Debounce raw_state into self._state / self._reason.

        Returns True iff self._state actually changed this call.

        ASYMMETRY, and this is deliberate, not an oversight: entering
        SAFE_TO_CROSS requires HYSTERESIS_FRAMES consecutive frames to
        agree, like every other transition -- a single flickered green
        detection must not immediately tell someone to cross. But LEAVING
        SAFE_TO_CROSS for anything else happens on the very next
        disagreeing frame, with no vote at all. Hysteresis exists to stop
        flicker from being annoying; it must never make the system slow to
        withdraw a safety claim. If a vehicle enters frame the instant
        after we said SAFE, waiting 8 more frames (~0.3s, but the vehicle
        arriving fast makes even that costly) before saying WAITING is
        exactly the wrong trade -- the one state where being slow to
        change is dangerous is the one we're leaving, not the one we're
        entering.
        """
        if raw_state == self._state:
            # Already agreeing -- nothing pending, nothing changed.
            self._vote_state = None
            self._vote_count = 0
            return False

        if self._state == SAFE_TO_CROSS:
            # Leaving SAFE: immediate, no vote (see docstring above).
            self._state = raw_state
            self._reason = raw_reason
            self._vote_state = None
            self._vote_count = 0
            return True

        # Entering/changing between any other pair of states: normal vote.
        if self._vote_state == raw_state:
            self._vote_count += 1
        else:
            self._vote_state = raw_state
            self._vote_count = 1

        if self._vote_count >= HYSTERESIS_FRAMES:
            self._state = raw_state
            self._reason = raw_reason
            self._vote_state = None
            self._vote_count = 0
            return True

        return False
