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

ADDED RULE NOT IN PLAN.md's ORIGINAL TABLE: a vehicle turning into the
crossing (safety.py's VehicleTracker.crossing_vehicles()) forces WAITING
with its own reason string, checked right after the red/vehicle-green
signal rules and before anything that can emit SAFE. PLAN.md's Phase 6
table only reasons about
min_ttc (a head-on closing-distance model), which cannot see a turning
vehicle's real risk -- see safety.py's module docstring for why a turning
vehicle can have a weak or absent TTC while still being the most dangerous
case for a pedestrian. This is exactly the kind of real gap PLAN.md's rule
table did not anticipate, so it is called out here rather than silently
folded into an existing rule's reason string.

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
ped_signal_green -- the two classes whose presence pushes toward SAFE -- are
only trusted above POSITIVE_EVIDENCE_MIN_CONF; every class that only ever
pushes toward WAITING (ped_signal_red, veh_signal_red, veh_signal_green) is
trusted at any confidence the detector already passed, because a
false-positive there only ever produces extra, harmless caution, never a
false SAFE. The try/except in update() is the hook a future camera-tilt
signal would raise into.

POST-REVIEW FIX (signal taxonomy split): this file used to have one
signal_red/signal_green pair shared between PEDESTRIAN signals (walking-man
icon, means "you may walk") and VEHICLE traffic lights (means "cars may
go"), and rule 5 below let ANY signal_green license SAFE_TO_CROSS. A
reviewer correctly flagged this as a false-safe path: four of the five
signal datasets this project trains on are vehicle dashcam shots (see
scripts/download_signal_datasets.py's docstring), so "signal_green" mostly
meant "the traffic light facing the CARS turned green", i.e. traffic
about to accelerate through the junction -- exactly the moment a pedestrian
must NOT be told it is safe to cross. The classes are now split into
ped_signal_red/ped_signal_green (pedestrian signal) and veh_signal_red/
veh_signal_green (vehicle signal). Only ped_signal_green can ever produce
SAFE_TO_CROSS. veh_signal_green is now treated as its own WAITING-only
danger signal (rule 2b below), checked before the rule that can emit SAFE,
so it can never be bypassed. veh_signal_red is explicitly NOT sufficient
evidence for SAFE on its own (stopped vehicles say nothing about turning
traffic, a stale observation, or a second carriageway) -- it is WAIT
evidence like ped_signal_red, and neither of them alone unlocks rule 5's
green check. See data/LABELLING.md section 2 for the full writeup.

LEGACY WEIGHTS: models/best.pt (the currently-deployed model, trained before
this split) still emits the OLD class names "signal_green"/"signal_red" and
will keep doing so until it is retrained on the new 12-class taxonomy. A
legacy "signal_green" detection must NEVER be able to reach SAFE_TO_CROSS --
the whole point of this fix is that a bare "green" cannot be trusted as
pedestrian permission. The conservative reading, given the training data was
dominated by vehicle-light datasets, is to treat a legacy "signal_green" as
if it were veh_signal_green (WAIT evidence, not SAFE evidence), and a legacy
"signal_red" as WAIT evidence too (folded in with ped_signal_red/
veh_signal_red). See _decide() below and
tests/test_fsm.py::test_legacy_signal_green_from_old_weights_never_produces_safe.
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
# presence pushes toward SAFE (crosswalk, ped_signal_green). Every
# WAIT-only signal class (ped_signal_red, veh_signal_red, veh_signal_green,
# and legacy signal_red/signal_green from old weights) is deliberately NOT
# gated by this -- see _decide().
POSITIVE_EVIDENCE_MIN_CONF = 0.6

# --- Legacy class-name compatibility (old models/best.pt) -----------------
# models/best.pt was trained before the ped_signal_*/veh_signal_* split and
# still emits the bare old names "signal_green"/"signal_red". It stays in
# use until a retrain happens, so this module must keep recognising those
# names -- and must map them to the CONSERVATIVE side of the new taxonomy,
# never to anything that can unlock SAFE. The old training data was
# dominated by vehicle dashcam datasets (see scripts/download_signal_
# datasets.py's docstring), so a legacy "signal_green" is read as
# veh_signal_green (WAIT evidence, danger: cars have right of way), and a
# legacy "signal_red" is read as generic WAIT evidence alongside
# ped_signal_red/veh_signal_red. See the module docstring's "LEGACY
# WEIGHTS" section for the full reasoning.
LEGACY_GREEN_WAIT_NAMES = ("veh_signal_green", "signal_green")
# veh_signal_red is deliberately NOT here. A red light for cars is the
# NORMAL companion of a green walk signal: at any working signalled
# crossing, when the pedestrian signal goes green the vehicle signal is
# red. Treating veh_signal_red as a WAIT trigger made ped_signal_green +
# veh_signal_red resolve to WAITING, i.e. it made SAFE unreachable at a
# real intersection even with a perfect detector. veh_signal_red is
# therefore NEUTRAL: it never licenses SAFE on its own (rule 5 still
# demands ped_signal_green), and it never blocks one either.
#
# A legacy bare "signal_red" from the old 10-class weights stays on the
# WAIT side, because with those weights there is genuinely no way to tell
# whether it was a pedestrian red (do not walk) or a vehicle red (cars
# stopped). Unknown means conservative.
LEGACY_RED_WAIT_NAMES = ("ped_signal_red", "signal_red")

# A crosswalk detection within this fraction of frame-half-width either side
# of centre reads as "centre" rather than left/right -- without a dead band
# a crosswalk sitting exactly on the centreline would flicker between
# "slightly left" and "slightly right" on pixel noise alone.
OFFSET_CENTRE_BAND = 0.1

# --- signal_countdown (class 9) -- deliberately NOT read anywhere below ---
# PLAN.md's class list has a signal_countdown class for numeric pedestrian
# countdown timers (common at Indian crossings), but as of adding it there
# is zero training data for it anywhere (see data/LABELLING.md section 8),
# so no detection of this class can be trusted yet, and this module reads
# none of its detections. Per data/LABELLING.md section 8: a countdown is
# information, not permission -- it must never be used to grant SAFE, and a
# countdown running down next to signal_green must never be read as
# extending the time _road_is_clear()/_required_crossing_time() considers
# safe. If a real, tested use is ever added, the only safe shape for it is
# STRICTLY MORE CONSERVATIVE than today's rules, e.g. refusing to (re-)enter
# SAFE_TO_CROSS when a countdown is detected very close to zero (the phase
# is about to change) -- never a rule that grants or extends SAFE based on
# a countdown value. With no labelled data to test either the detector or
# such a rule against, it stays a comment, not code.


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
        """PLAN.md's five rules, in order, plus two added rules inserted
        before the TTC check so neither can ever be bypassed by the one rule
        that emits SAFE: rule 2b (a green VEHICLE signal is WAIT evidence,
        see module docstring's "POST-REVIEW FIX") and rule 3 (turning
        vehicles, see module docstring). Returns
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

        # Rule 2: a PEDESTRIAN red (or an ambiguous legacy red) -> WAITING,
        # unconditionally. No confidence gate here on purpose (see module
        # docstring): a false-positive red only ever produces extra caution,
        # so there is no safety reason to demand a higher bar for it, and
        # demanding one would risk missing a real red. This check happens
        # BEFORE min_ttc or any green signal is even looked at, so nothing
        # below can override an explicit "do not walk".
        #
        # veh_signal_red is NOT checked here -- see LEGACY_RED_WAIT_NAMES.
        # It is the normal companion of a green walk signal, so blocking on
        # it would make SAFE unreachable at a real crossing. It still cannot
        # license SAFE: rule 5 demands ped_signal_green, and stopped cars
        # are not permission to walk.
        if any(d.cls_name in LEGACY_RED_WAIT_NAMES for d in detections):
            return WAITING, "wait, signal is red", tracker.min_ttc(), offset, direction

        # Rule 2b (added, post-review fix -- see module docstring's
        # "POST-REVIEW FIX" section): a green VEHICLE signal (or a legacy
        # "signal_green" from old weights, conservatively read as a vehicle
        # signal) -> WAITING, unconditionally, same no-confidence-gate
        # reasoning as rule 2. This is the heart of the false-safe fix: a
        # green light for CARS means traffic has right of way, which is
        # positive evidence of danger to a pedestrian, not permission to
        # cross. This must be checked BEFORE rule 5 (the only rule that can
        # emit SAFE) so it can never be bypassed by a coincidentally-present
        # ped_signal_green in the same frame -- if both are visible, that is
        # a scene that needs a human's judgement, not this system's, so it
        # resolves to the conservative side.
        if any(d.cls_name in LEGACY_GREEN_WAIT_NAMES for d in detections):
            return WAITING, "wait, vehicle signal is green", tracker.min_ttc(), offset, direction

        # Rule 3 (added, not in PLAN.md's original five-rule table -- see
        # module docstring): a vehicle turning into the crossing. This has
        # to sit here, after signal_red and before anything that can emit
        # SAFE, so nothing downstream can ever override it. safety.py's
        # crossing_vehicles() is a separate signal from min_ttc(): a
        # turning vehicle typically has WEAK box growth while it swings
        # through the turn (see safety.py's module docstring), so it can
        # have no TTC at all, or even read NOT_APPROACHING on growth alone,
        # while still being the single most dangerous case for a
        # pedestrian. It must not be treated as safe just because TTC has
        # nothing to say about it.
        if tracker.crossing_vehicles():
            return (
                WAITING,
                "wait, vehicle turning across the crossing",
                tracker.min_ttc(),
                offset,
                direction,
            )

        min_ttc = tracker.min_ttc()

        # Rule 4: a vehicle is close enough that PLAN.md's flat 5.0s
        # threshold alone says wait, regardless of anything else.
        if min_ttc is not None and min_ttc < DECISION_TTC_THRESHOLD:
            return WAITING, "wait, vehicle approaching", min_ttc, offset, direction

        # Rule 5: PEDESTRIAN green signal AND a road we can honestly call
        # clear. Only ped_signal_green may license SAFE_TO_CROSS -- this is
        # the one and only rule in this whole file that can return
        # SAFE_TO_CROSS, and it is gated at POSITIVE_EVIDENCE_MIN_CONF (see
        # module docstring's confidence-asymmetry note). veh_signal_green
        # (and legacy signal_green) can never reach here: rule 2b above
        # already returned WAITING for them. veh_signal_red alone is also
        # not looked at here at all -- it is WAIT evidence (rule 2), never
        # SAFE evidence, so its mere absence is not treated as permission
        # either; only an actual ped_signal_green detection can produce SAFE.
        ped_signal_green = _best_detection(detections, "ped_signal_green", POSITIVE_EVIDENCE_MIN_CONF)
        if ped_signal_green is not None and self._road_is_clear(min_ttc, tracker.unresolved_vehicles()):
            return SAFE_TO_CROSS, "safe to cross now", min_ttc, offset, direction

        # Rule 6: otherwise -- e.g. no pedestrian signal detected at all,
        # ped_signal_green too low-confidence to trust, or the road fails
        # the crossing-time or unresolved-vehicle check. PLAN.md's default
        # is always "wait".
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
