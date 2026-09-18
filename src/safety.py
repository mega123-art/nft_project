"""
Phase 5: motion and time-to-contact.

A single camera has no depth sensor. The only cue we get for "is this
vehicle approaching" is that its bounding box grows in the image as it gets
closer. This module turns that growth into a rough time-to-contact (TTC)
estimate per tracked vehicle, and exposes the smallest one across the scene.

Derivation of the TTC formula (explain this in the viva):

    Pinhole projection: a real object of width W at distance D projects to
    an image width w = f * W / D, where f is the camera's focal length.
    W and f are constants for a given vehicle and camera, so:

        w = k / D            (k = f * W, a constant)

    We want distance-over-closing-speed, D / (-dD/dt). We could get there by
    differentiating w directly, but w = k/D is a HYPERBOLA in time whenever
    D changes at a constant rate (a constant-velocity approach) -- it curves,
    getting steeper as the vehicle closes. Fitting a straight line to a
    handful of points on a curve underestimates the slope at the newest
    (steepest) point, which underestimates dw/dt, which OVERESTIMATES
    TTC = w / (dw/dt). That is a smoothing artifact, and worse, it is
    optimistic exactly when the vehicle is closest -- the one place this
    system is not allowed to be optimistic.

    The fix is to work with the RECIPROCAL of the width instead. Since
    w = k/D, we have:

        u = 1/w = D/k

    D/k is just distance scaled by a constant, so u is LINEAR in time
    whenever distance is linear in time (constant closing speed) --
    it is not a curve to approximate, it is exactly the right shape for a
    straight-line fit:

        u(t) = D(t)/k = (D0 - v*t) / k          (D0 = distance at t=0)
        du/dt = -v/k                             (a constant)

    Rearranging D/v (= TTC) in terms of u and du/dt:

        u / (-du/dt) = (D/k) / (v/k) = D/v = TTC

    So TTC = u / (-du/dt): the fitted reciprocal-width divided by the
    negative of its slope. A vehicle getting closer means D is shrinking, so
    u = D/k is shrinking too, so du/dt is NEGATIVE while the vehicle
    approaches -- "approaching" means a falling 1/width, not a falling
    width. This sign is the easiest place in this file to get backwards, so
    it is spelled out again at each use below. No focal length or real
    vehicle width needed, which is the whole point -- we never calibrated
    either. This is still only an estimate: it assumes constant closing
    speed over the window and a head-on approach. Under exactly that
    assumption there is no smoothing lag any more (see
    test_approaching_vehicle_ttc_matches_analytic_closely). If the vehicle
    is accelerating -- speeding up as it closes, which is common in real
    traffic -- the constant-velocity assumption breaks and the fitted line
    lags the true (now curving) 1/width trend, which makes the reported TTC
    OPTIMISTIC again (it reports more time than truly remains; see
    test_accelerating_vehicle_ttc_is_optimistic). Phase 6 must not treat a
    positive TTC margin over its threshold as exact -- keep a margin of its
    own on top of this number.

IMPORTANT -- None is ambiguous, and the caller must not read it as "safe".
A missing TTC can mean two very different things: (a) the vehicle is
genuinely not a threat (a real assessment: receding, or parked with a full
window of near-flat data), or (b) we simply do not know yet (too few
samples, or a degenerate fit). PLAN.md's Phase 6 rule only fires WAITING
when min_ttc() is not None and < 5.0 -- so case (b) on a vehicle that is
actually bearing down on the user would silently fail to trigger WAITING,
and nothing else would stop a false SAFE. That is exactly the one
unacceptable failure mode this whole project exists to avoid.

But collapsing everything-but-APPROACHING into "unresolved" is also wrong,
for a different reason: at a real crossing there is almost always a parked
vehicle, a stopped bus, or a stationary rickshaw somewhere in frame. If a
merely-parked vehicle counted as unresolved forever, the system would never
be allowed to say SAFE at all -- and a system that is permanently stuck on
"wait" teaches the user to stop listening to it, which is its own kind of
safety failure. Both failure modes (false SAFE, and useless permanent WAIT)
have to be avoided, not just one.

So every track carries one of three explicit statuses (module constants
below), not just a TTC-or-None:

    APPROACHING     -- fit succeeded over a full window and says closing.
                       A real TTC is available.
    NOT_APPROACHING -- fit succeeded over a full window and says receding,
                       parked, or flat. This IS a real assessment, backed
                       by enough data, and it counts as evidence of safety
                       for that one vehicle.
    UNKNOWN         -- we cannot say anything yet: fewer than
                       MIN_SAMPLES_FOR_TTC samples, a degenerate fit, or a
                       bad (non-positive) box width. Not evidence of
                       anything, in either direction.

The ordering that MUST hold in the code: sample count and fit validity are
checked FIRST, before the growth rate is even looked at. A track can only
reach NOT_APPROACHING through a successful fit over a full window -- it
must never fall out of "insufficient data" straight into "assessed as
safe". That would recreate exactly the false-SAFE gap above, just one layer
deeper (an unassessed approaching vehicle silently reading as
NOT_APPROACHING instead of silently reading as no-TTC).

DECIDING APPROACHING vs NOT_APPROACHING -- why a single fixed threshold on
approach_rate doesn't work: box-edge jitter of about JITTER_PIXELS pixels
in the width w introduces noise in u = 1/w of roughly
delta/w^2 (from d(1/w) = -dw/w^2). Spread over a window of duration T, the
apparent approach_rate that pure noise alone can fake is therefore roughly

    rate_noise ~= delta / (w^2 * T)

This depends on w. A small, distant box has a much noisier apparent
approach_rate than a large, close one for the exact same pixel jitter --
so any single constant threshold on approach_rate is simultaneously too
strict for distant vehicles (rejecting real slow approaches as noise,
which is the dangerous direction -- a real approach misread as
NOT_APPROACHING is misread as safety evidence) and too loose for close
ones (letting jitter flicker the status). This module therefore does not
threshold approach_rate directly (beyond a cheap MIN_APPROACH_RATE guard
against zero/negative rates). Instead it converts the noise floor into a
TTC "noise horizon": rearranging TTC = u/approach_rate with
approach_rate ~= rate_noise gives

    TTC_horizon = w * T / delta

Any measured TTC longer than this horizon is indistinguishable from noise
on a static box, given the assumed jitter. So:

  - TTC <= TTC_horizon: the closing rate is bigger than what jitter alone
    could produce -- trust it. Status APPROACHING.
  - TTC >  TTC_horizon: either this is jitter on a static/receding box, or
    it is a real approach so distant that its TTC is far beyond anything
    decision-relevant anyway. Status NOT_APPROACHING -- SAFE, provided the
    horizon itself is comfortably above PLAN.md's 5.0s WAITING threshold
    (see HORIZON_SAFETY_MARGIN). This is self-correcting: as a real
    vehicle closes, w grows, the horizon grows, and its TTC keeps falling,
    so it flips to APPROACHING well before its TTC nears 5s.
  - If the horizon itself is too close to the 5s decision threshold to
    trust a NOT_APPROACHING call beyond it (a box so small the horizon
    could plausibly sit right where a real approach would also need to be
    flagged), the honest answer is UNKNOWN, not NOT_APPROACHING -- we
    genuinely cannot rule out a real threat at that box size with this
    much jitter.

The smoothed width from the fit (1/smoothed_u), not the raw latest width,
feeds the horizon calculation -- otherwise the horizon itself would be
noisy, defeating the point.

KNOWN LIMITATION, found while re-running the synthetic harness against this
scheme -- reported here rather than hidden: for a small/distant, fast-
closing vehicle, the horizon itself starts small (it scales with w), so
"UNKNOWN, beyond the horizon but below HORIZON_SAFETY_MARGIN" can persist
right up until the vehicle's TTC has fallen to nearly the horizon's own
(small) value -- which can be close to, or even past, the 5s decision
threshold by the time APPROACHING first fires with a number. Concretely,
in one synthetic run (D0=60m, v=12m/s closing, starting box ~15px wide),
the status stayed UNKNOWN all the way from true TTC=5.0s down to true
TTC=4.08s, and only reported APPROACHING once true TTC had already reached
4.0s. This is NOT a false-SAFE: the status is UNKNOWN throughout, never
NOT_APPROACHING, so unresolved_vehicles() correctly keeps blocking SAFE the
whole time. But it means the "APPROACHING with a number" signal itself can
arrive later than PLAN.md's eyeball-it "does the number visibly count down
before the vehicle gets there" check might expect, for vehicles that start
out small in the frame. Whether this matters in practice depends on real
box sizes at decision-relevant distances, which are unknown until Phase 4
footage and a calibrated camera exist. If it turns out to matter, the fix
is not a smaller MIN_APPROACH_RATE (that reintroduces the original
false-SAFE risk) but likely a smaller, measured JITTER_PIXELS (less
assumed noise needs a smaller horizon margin) or a longer history window.

VehicleTracker.unresolved_vehicles() counts only currently-visible UNKNOWN
tracks -- not NOT_APPROACHING ones. A parked car settles into
NOT_APPROACHING after its first few frames and stops blocking SAFE; an
approaching vehicle that hasn't been watched long enough stays UNKNOWN and
keeps blocking it. CALLER CONTRACT: Phase 6 must never declare SAFE while
unresolved_vehicles() is non-zero.

LATERAL CROSSING -- turning vehicles, not just approaching/receding ones.
The most dangerous real case for a pedestrian is a vehicle turning INTO the
crossing, not one driving straight at the camera. lateral_drift (the signed
px/s slope of box-centre movement, computed below) already carries that
signal, but a raw px/s number is meaningless on its own: 50px/s of drift is
"crossing our path fast" for a box 40px wide and "barely moving" for a box
400px wide, because the same real-world sideways speed produces a bigger
apparent drift the closer (bigger) the box is. So we normalise by the
vehicle's own (mean) box width, giving crossing_ratio -- a scale-invariant
"fraction of its own width moved sideways per second". The same real-world
turn reads the same whether the vehicle is near or far, small or large.

This needs the same false-alarm guard TTC already has, for the same reason
(a parked vehicle's box-centre jitters by a pixel or two same as its box
edges do, and that must not read as "crossing" forever). Reusing
JITTER_PIXELS (not inventing a second, unmeasured constant) keeps the
assumption consistent across the module: perturbing a centre measurement by
JITTER_PIXELS over a window of duration T can fake an apparent lateral
speed of about JITTER_PIXELS / T (no squaring here, unlike TTC_horizon --
we differentiate the raw centre position directly, not its reciprocal).
Normalised by width the same way as the real signal, the noise floor on
crossing_ratio is

    ratio_noise ~= JITTER_PIXELS / (T * mean_width)

A measured crossing_ratio must clear CROSSING_NOISE_MARGIN times this floor
before TrackState.is_crossing is allowed to be True. Below
MIN_SAMPLES_FOR_TTC samples (same gate as _assess()), crossing_ratio is None
and is_crossing is False -- "not enough data" is not "not crossing", but
unlike the TTC path there is no separate UNKNOWN-style tri-state exposed
here: is_crossing is a plain bool that VehicleTracker.crossing_vehicles()
reads for currently-visible tracks, and fsm.py treats "no crossing detected
yet" the same as "not crossing" for this specific signal, because the
default output is always WAITING anyway (PLAN.md's ground rule) -- an
under-confident crossing flag never produces a false SAFE by itself, it
just means one fewer reason to refuse SAFE. What would be unsafe is the
reverse (flagging a merely jittering parked car as crossing forever and
never letting SAFE happen), which is exactly what the noise floor guards.

INTERACTION WITH APPROACHING / NOT_APPROACHING / UNKNOWN -- this is the
part that actually required changing _assess(), not just adding a new
field next to it. A vehicle turning into the crossing typically has a WEAK
box-growth signal while it swings through the turn (it is moving mostly
sideways, not toward the camera), which is exactly the signal _assess()'s
early guard (approach_rate < MIN_APPROACH_RATE) and its beyond-horizon
branch both read as "NOT_APPROACHING -- a real assessment, safe". Before
this change, a turning vehicle with real lateral drift but weak forward
growth would sail straight into NOT_APPROACHING and get treated as
evidence of safety by fsm.py's _road_is_clear() -- precisely the
false-safe gap this whole addition exists to close. So both places
_assess() would otherwise return NOT_APPROACHING now check is_crossing
first: if the track is independently flagged as crossing, the honest
answer is UNKNOWN (we cannot call this vehicle safe on the growth signal
alone, and we are not claiming it is definitely a threat by that signal
either -- crossing_vehicles() is what actually forces the FSM's hand, not
this status). A track that is both approaching (real box growth) AND
crossing keeps its normal APPROACHING/ttc status unchanged -- crossing_ratio
is independent, additive information, not a replacement for the TTC
assessment; fsm.py is expected to treat "approaching AND crossing" as
worse than either alone, which it does simply by having both signals
available to check.

CALLER CONTRACT: fsm.py must treat any vehicle in
VehicleTracker.crossing_vehicles() as forcing WAITING, regardless of what
min_ttc() or unresolved_vehicles() say -- a turning vehicle's danger is not
captured by TTC at all (TTC models a head-on closing distance; a turning
vehicle's real risk is that its path will intersect the crossing, which
lateral drift is the only signal in this module that speaks to).

Track history keyed by track_id holds the last few (timestamp, width)
samples. Differentiating two raw, noisy box-edge measurements directly is
exactly what PLAN.md warns against -- a one-pixel jitter in a fast-moving
window turns into a huge, spurious derivative. Instead we fit a straight
line (ordinary least squares) through the whole window of *reciprocal*
widths against time. The slope of that line is a smoothed du/dt -- it uses
every point in the window rather than just the two noisiest ones (the
endpoints), and the fitted value at the latest timestamp is a smoothed
reciprocal-width to divide by. This is the "fit a slope over the window"
option PLAN.md allows, and it is simple enough to explain line by line.
"""

from collections import deque
from typing import Optional

# Per-track status. Plain string constants -- no enum module needed, and a
# string prints readably in a debugger or an overlay without extra work.
# See the module docstring for the full explanation of each one and the
# ordering rule that keeps UNKNOWN from leaking into NOT_APPROACHING.
APPROACHING = "APPROACHING"
NOT_APPROACHING = "NOT_APPROACHING"
UNKNOWN = "UNKNOWN"

# Only these classes are ever treated as a collision threat. A person
# approaching fast is not a vehicle threat -- pedestrians are handled by
# the FSM (Phase 6), not by TTC.
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "autorickshaw"}

HISTORY_LEN = 5  # samples kept per track, per PLAN.md

# A track that hasn't been seen for this many consecutive update() calls is
# dropped -- it left the frame (or the tracker lost it), and stale history
# must not keep contributing to min_ttc().
MAX_MISSED_FRAMES = 15

# We fit reciprocal width (u = 1/width) against time. Approaching means u
# is FALLING (du/dt < 0, see the module docstring's sign discussion). We
# call the positive, "how fast is it approaching" quantity the approach
# rate: approach_rate = -du/dt.
#
# This is deliberately just a cheap early guard against zero/negative rates
# (a shrinking or perfectly flat box), not the mechanism that separates
# real approaches from noise -- seeing MIN_APPROACH_RATE alone as "the"
# noise filter was tried and rejected: a single fixed threshold on
# approach_rate is too strict for a small/distant box and too loose for a
# large/close one for the same pixel jitter (see the module docstring's
# noise-horizon derivation, which does the real work via TTC_horizon
# below). Kept low and near-zero on purpose.
MIN_APPROACH_RATE = 1e-4  # (1/pixels) per second, i.e. -du/dt threshold

# The one tunable in the noise-horizon scheme: an assumption about how many
# pixels a detector's box edge jitters frame to frame on a stable, unmoving
# vehicle. Not measured against our actual model yet (models/best.pt is
# domain-mismatched on the available sample clips -- see PLAN.md), so this
# is an estimate pending real measurement once Phase 4's footage exists;
# 1.5px is a reasonable starting assumption for a YOLO box edge, and errs
# slightly generous (assumes a bit more jitter than a well-trained detector
# should have) rather than optimistic.
JITTER_PIXELS = 1.5

# PLAN.md's Phase 6 rule fires WAITING when TTC < 5.0s. Referenced here only
# to size the safety margin on the noise horizon below -- this module makes
# no decision at 5.0s itself.
DECISION_TTC_THRESHOLD = 5.0

# A NOT_APPROACHING call beyond the noise horizon is only trustworthy if
# the horizon itself is comfortably clear of the decision threshold above
# -- otherwise a box so small that horizon and threshold nearly coincide
# could hide a real approach whose TTC is right around 5s. 2x is a plain,
# explainable margin: "the horizon must be at least twice the distance we
# actually make decisions at."
HORIZON_SAFETY_MARGIN = 2.0

# The lateral-crossing analogue of HORIZON_SAFETY_MARGIN: how many multiples
# of the jitter noise floor a measured crossing_ratio must clear before it
# is trusted as real sideways motion rather than box-centre jitter on a
# parked vehicle (see module docstring's crossing_ratio derivation). 3x is a
# plain, explainable margin in the same spirit as HORIZON_SAFETY_MARGIN's
# 2x -- there is no measured false-positive rate to tune this against yet
# (same caveat as JITTER_PIXELS itself), so it errs toward requiring a
# clearer signal rather than a marginal one.
CROSSING_NOISE_MARGIN = 3.0

# A fit through 2-3 points is exactly the noisy, unstable regime the
# accelerating/noisy tests above show spurious readings in -- a brand new
# track must not report a TTC (or, just as bad, a confident-looking None
# that a caller mistakes for "not a threat") until it has enough samples
# for the line fit to mean something. At 8-15 fps this delays a new
# vehicle's first TTC by roughly 0.2-0.4s; that delay is only acceptable
# because unresolved_vehicles() (see VehicleTracker) makes "not enough data
# yet" visible to the caller instead of silently looking like "no threat".
MIN_SAMPLES_FOR_TTC = 4


def _linear_fit(xs, ys):
    """Ordinary least squares slope + intercept of ys against xs.

    Returns (slope, intercept), or None if there are fewer than two
    distinct x values (a single observation, or several observations with
    the exact same timestamp -- both would divide by zero below).
    """
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if denominator <= 0:
        # every timestamp in the window is identical -- no elapsed time to
        # differentiate over.
        return None

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


class TrackState:
    """Per-vehicle history and the derived quantities the overlay/FSM read."""

    def __init__(self, cls_name):
        self.cls_name = cls_name
        self.widths = deque(maxlen=HISTORY_LEN)
        self.timestamps = deque(maxlen=HISTORY_LEN)
        self.centres = deque(maxlen=HISTORY_LEN)
        self.frames_since_seen = 0

        # Recomputed on every update() while the track is visible; read by
        # the overlay and, later, the FSM. status is one of APPROACHING /
        # NOT_APPROACHING / UNKNOWN (see module docstring); ttc is only
        # ever a number when status == APPROACHING.
        self.status = UNKNOWN
        self.ttc = None
        self.lateral_drift = None  # signed px/s of box-centre movement

        # Scale-invariant lateral crossing signal -- see module docstring.
        # crossing_ratio is the signed, width-normalised drift once assessed
        # (None while there isn't enough history to trust it); is_crossing
        # is the plain bool VehicleTracker.crossing_vehicles() reads, False
        # both when genuinely not crossing and when not yet assessed (see
        # docstring for why an under-confident False here is the safe
        # direction for this particular signal).
        self.crossing_ratio = None
        self.is_crossing = False

    def add_observation(self, width, centre_x, timestamp):
        self.widths.append(width)
        self.centres.append(centre_x)
        self.timestamps.append(timestamp)
        self.frames_since_seen = 0
        self.lateral_drift = self._compute_lateral_drift()
        # Crossing must be assessed BEFORE _assess() below -- _assess()
        # reads self.is_crossing to stop a turning vehicle's weak forward
        # growth from being misread as NOT_APPROACHING (see docstring).
        self.crossing_ratio, self.is_crossing = self._assess_crossing()
        self.status, self.ttc = self._assess()

    def _assess(self):
        """Return (status, ttc) for the current window.

        Ordering matters and must not be reshuffled: sample count and fit
        validity are checked FIRST. Only once a fit has actually succeeded
        over a full window do we ask whether it says approaching or not.
        This is what stops "not enough data" from ever being read as
        NOT_APPROACHING (see module docstring).
        """
        widths = list(self.widths)
        if len(widths) < MIN_SAMPLES_FOR_TTC:
            # Not enough history yet for the fit to be trustworthy.
            return UNKNOWN, None
        if any(w <= 0 for w in widths):
            # A non-positive box width is a bad detection, not real data --
            # 1/width would divide by zero or flip sign. Not an assessment.
            return UNKNOWN, None

        # See the module docstring: fit reciprocal width (u = 1/w) against
        # time, not width itself -- u is exactly linear in time under a
        # constant-velocity approach, so this fit has no smoothing lag,
        # unlike fitting width directly.
        reciprocal_widths = [1.0 / w for w in widths]
        fit = _linear_fit(list(self.timestamps), reciprocal_widths)
        if fit is None:
            # Degenerate fit (e.g. every timestamp identical) -- not an
            # assessment either way.
            return UNKNOWN, None
        u_slope, u_intercept = fit

        # u = 1/width is FALLING while the vehicle approaches (it's getting
        # closer, so width grows, so 1/width shrinks). approach_rate is the
        # positive-when-approaching quantity: the negative of that slope.
        approach_rate = -u_slope
        if approach_rate < MIN_APPROACH_RATE:
            # Cheap early guard only -- an outright zero or negative rate
            # (shrinking or perfectly flat) needs no noise-horizon
            # reasoning at all. Ordinarily a real assessment: not a threat.
            # BUT a vehicle turning into the crossing often has exactly this
            # weak/flat forward-growth signature while it swings sideways --
            # see module docstring. If lateral drift independently says this
            # track is crossing, "not a threat" would be a false-safe signal
            # handed to fsm.py, so refuse to assess rather than clear it.
            return self._not_approaching_unless_crossing()

        latest_t = self.timestamps[-1]
        smoothed_u = u_slope * latest_t + u_intercept
        if smoothed_u <= 0:
            # Degenerate fit (can happen with a very short, noisy window) --
            # refuse to report rather than return a nonsense number.
            return UNKNOWN, None

        ttc = smoothed_u / approach_rate
        # Smoothed width, from the same fit -- not the raw latest width --
        # so the horizon below isn't itself noisy (see module docstring).
        smoothed_width = 1.0 / smoothed_u

        window_duration = self.timestamps[-1] - self.timestamps[0]
        if window_duration <= 0:
            # Shouldn't happen once _linear_fit has succeeded (that already
            # requires non-identical timestamps), but guard explicitly
            # rather than assume.
            return UNKNOWN, None

        # See module docstring for the derivation: any TTC beyond this is
        # indistinguishable from pixel jitter on a static box, given the
        # assumed JITTER_PIXELS of edge noise.
        ttc_horizon = smoothed_width * window_duration / JITTER_PIXELS

        if ttc <= ttc_horizon:
            # The apparent closing rate is bigger than jitter alone could
            # produce at this box size and window -- trust it.
            return APPROACHING, ttc

        # Beyond the horizon: either this is jitter on a static/receding
        # box, or a real approach so distant its TTC is far beyond
        # anything decision-relevant. Only call it NOT_APPROACHING if the
        # horizon itself sits comfortably clear of PLAN.md's 5s decision
        # threshold -- otherwise we cannot honestly rule out a real,
        # decision-relevant approach at this box size.
        if ttc_horizon >= HORIZON_SAFETY_MARGIN * DECISION_TTC_THRESHOLD:
            # Same reasoning as the early guard above: a distant/slow
            # apparent approach that's within noise could still be a
            # turning vehicle whose sideways motion is the real signal.
            return self._not_approaching_unless_crossing()
        return UNKNOWN, None

    def _not_approaching_unless_crossing(self):
        """Shared by both places _assess() would otherwise say
        NOT_APPROACHING. See module docstring: a turning vehicle can look
        like "not approaching" on box growth alone while genuinely crossing
        our path, and that must never read as safety evidence."""
        if self.is_crossing:
            return UNKNOWN, None
        return NOT_APPROACHING, None

    def _compute_lateral_drift(self):
        fit = _linear_fit(list(self.timestamps), list(self.centres))
        if fit is None:
            return None
        drift_rate, _intercept = fit
        return drift_rate

    def _assess_crossing(self):
        """Return (crossing_ratio, is_crossing) for the current window.

        See module docstring for the crossing_ratio derivation and the
        noise-floor guard. Same MIN_SAMPLES_FOR_TTC/positive-width gate as
        _assess(), for the same reason: a fit through too few or bad points
        is not a real assessment.
        """
        widths = list(self.widths)
        if len(widths) < MIN_SAMPLES_FOR_TTC or any(w <= 0 for w in widths):
            return None, False

        fit = _linear_fit(list(self.timestamps), list(self.centres))
        if fit is None:
            return None, False
        drift_rate, _intercept = fit

        window_duration = self.timestamps[-1] - self.timestamps[0]
        if window_duration <= 0:
            return None, False

        # Mean width over the window, not just the latest sample, so one
        # noisy frame can't swing the normalisation -- same smoothing
        # intent as the TTC path's smoothed_width.
        mean_width = sum(widths) / len(widths)

        crossing_ratio = drift_rate / mean_width

        # Noise floor: see module docstring. A JITTER_PIXELS wobble in the
        # measured centre over this window could fake an apparent lateral
        # speed of about JITTER_PIXELS / window_duration; normalised by
        # width the same way as the real signal above.
        noise_floor_ratio = JITTER_PIXELS / (window_duration * mean_width)

        is_crossing = abs(crossing_ratio) > CROSSING_NOISE_MARGIN * noise_floor_ratio
        return crossing_ratio, is_crossing


class VehicleTracker:
    """Keeps per-track_id history and derives TTC / lateral drift from it."""

    def __init__(self):
        self.tracks = {}  # track_id -> TrackState

    def update(self, detections, timestamp):
        """Feed one frame's detections in. timestamp is a real elapsed-time
        clock (e.g. time.monotonic()), not a frame counter -- the pipeline
        runs at a variable 8-15 fps so a fixed dt assumption would be wrong.
        """
        seen_ids = set()

        for det in detections:
            if det.track_id is None:
                # Tracker hasn't confirmed an id for this box yet -- we have
                # nothing stable to key history on, so skip it. It will
                # start contributing once ByteTrack assigns it an id.
                continue
            if det.cls_name not in VEHICLE_CLASSES:
                # Only vehicles are collision threats (see module docstring).
                continue

            seen_ids.add(det.track_id)
            track = self.tracks.get(det.track_id)
            if track is None:
                track = TrackState(det.cls_name)
                self.tracks[det.track_id] = track
            track.cls_name = det.cls_name

            width = det.x2 - det.x1
            centre_x = (det.x1 + det.x2) / 2.0
            track.add_observation(width, centre_x, timestamp)

        # Age out tracks this frame didn't touch, and drop ones that have
        # been missing too long.
        dead_ids = []
        for track_id, track in self.tracks.items():
            if track_id in seen_ids:
                continue
            track.frames_since_seen += 1
            if track.frames_since_seen >= MAX_MISSED_FRAMES:
                dead_ids.append(track_id)
        for track_id in dead_ids:
            del self.tracks[track_id]

    def min_ttc(self):
        """Smallest TTC across all currently approaching vehicles, or None
        when nothing is approaching. None must be treated conservatively by
        the caller -- it means "no threat information", not "no threat".
        """
        ttcs = [t.ttc for t in self.tracks.values() if t.ttc is not None]
        if not ttcs:
            return None
        return min(ttcs)

    def get_ttc(self, track_id) -> Optional[float]:
        """Per-track TTC lookup for the overlay. None unless status == APPROACHING."""
        track = self.tracks.get(track_id)
        return track.ttc if track is not None else None

    def get_status(self, track_id) -> Optional[str]:
        """Per-track status (APPROACHING / NOT_APPROACHING / UNKNOWN) for the
        overlay and Phase 6. None if track_id isn't currently tracked."""
        track = self.tracks.get(track_id)
        return track.status if track is not None else None

    def get_crossing_ratio(self, track_id) -> Optional[float]:
        """Per-track normalised lateral crossing ratio, for the overlay.
        None unless the track has enough history to assess it (see
        TrackState._assess_crossing)."""
        track = self.tracks.get(track_id)
        return track.crossing_ratio if track is not None else None

    def crossing_vehicles(self):
        """Currently-visible tracks (frames_since_seen == 0) whose motion
        indicates they are crossing the user's path -- see module
        docstring's crossing_ratio derivation. Returns TrackState objects
        (not just a count) so a caller that wants more than "is anything
        crossing" (e.g. an overlay, or a future per-vehicle announcement)
        doesn't have to re-derive it.

        CALLER CONTRACT: fsm.py must treat a non-empty result as forcing
        WAITING, independent of min_ttc()/unresolved_vehicles() -- a
        turning vehicle's danger is not something TTC (built for head-on
        closing distance) can see at all.
        """
        return [
            track
            for track in self.tracks.values()
            if track.frames_since_seen == 0 and track.is_crossing
        ]

    def unresolved_vehicles(self):
        """Count of vehicles visible in the current frame with UNKNOWN status.

        UNKNOWN means "not enough data to say anything yet" -- it does NOT
        include NOT_APPROACHING, which is a real assessment (backed by a
        full window of data) that a vehicle is receding or parked. That
        distinction is the whole point: without it, an ordinary parked
        vehicle -- present at nearly every real crossing -- would count as
        unresolved forever and the system would never be allowed to say
        SAFE. A track that has left the frame this update
        (frames_since_seen > 0) is excluded: it is not currently visible,
        so it cannot be a vehicle we are failing to assess right now.

        CALLER CONTRACT: Phase 6 must never declare SAFE while this is
        non-zero. A visible, unassessed vehicle is not positive evidence of
        safety, and PLAN.md's ground rule is that only explicit positive
        evidence produces SAFE.
        """
        return sum(
            1
            for track in self.tracks.values()
            if track.frames_since_seen == 0 and track.status == UNKNOWN
        )
