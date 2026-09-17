"""
Phase 5 tests for src/safety.py.

These are synthetic. models/best.pt detects poorly on the sample clips in
data/raw_videos/ (they're elevated CCTV of non-Indian roads -- domain
mismatch, not a bug), so the sample clips can't be used to verify TTC
behaviour. Instead we synthesise Detection sequences directly and check the
tracker does what PLAN.md specifies. Each test encodes an *intended*
behaviour from PLAN.md/the Phase 5 spec, not just "whatever the code
currently does".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety import (
    VehicleTracker,
    VEHICLE_CLASSES,
    MAX_MISSED_FRAMES,
    MIN_APPROACH_RATE,
    MIN_SAMPLES_FOR_TTC,
    DECISION_TTC_THRESHOLD,
    APPROACHING,
    NOT_APPROACHING,
    UNKNOWN,
)
from detector import Detection


def make_detection(width, track_id=1, cls_name="car", centre_x=None, height=20.0):
    """Build a Detection with a given box width, centred at centre_x (or 0)."""
    if centre_x is None:
        x1, x2 = 0.0, width
    else:
        x1, x2 = centre_x - width / 2.0, centre_x + width / 2.0
    return Detection(cls_name=cls_name, conf=0.9, x1=x1, y1=0.0, x2=x2, y2=height, track_id=track_id)


def feed(tracker, widths, timestamps, track_id=1, cls_name="car"):
    """Push a sequence of (width, timestamp) pairs for one track and return
    the list of tracker.min_ttc() readings taken after each push."""
    readings = []
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=track_id, cls_name=cls_name)], t)
        readings.append(tracker.min_ttc())
    return readings


# ---------------------------------------------------------------------------
# Approaching vehicle: TTC should fall over time and track the analytic value
# ---------------------------------------------------------------------------

def test_approaching_vehicle_ttc_decreases_and_matches_analytic():
    # Physical setup: a vehicle closes at a constant real-world speed v,
    # starting at distance D0. Distance is linear in time:
    #     D(t) = D0 - v*t
    # The pinhole model used in safety.py says image width w = k / D for
    # some constant k (k = focal_length * real_vehicle_width). So:
    #     w(t) = k / (D0 - v*t)
    # safety.py fits 1/w (not w) against time, because 1/w = D/k is exactly
    # linear in time under this constant-velocity assumption -- see the
    # module docstring. The analytically exact remaining time-to-contact at
    # any instant is simply true_ttc(t) = D(t) / v.
    # k is chosen large enough that the box is a realistic tens-of-pixels
    # size even at the start of the window -- a box that starts out only
    # ~10px wide sits below the noise-horizon safety margin (see
    # HORIZON_SAFETY_MARGIN in safety.py) and correctly reports UNKNOWN
    # rather than a number for its first couple of readings. That's a real,
    # documented, and safe (never a false SAFE) limitation of the
    # noise-horizon scheme for very small/distant boxes -- not a bug in
    # this test. k=4000 keeps the synthetic box comfortably above that
    # regime throughout, so this test can focus on TTC accuracy instead.
    D0 = 60.0
    v = 10.0  # m/s closing speed -> contact at t = D0/v = 6.0s
    k = 4000.0
    dt = 0.1  # 10 fps, inside the pipeline's stated 8-15 fps range

    tracker = VehicleTracker()
    n_samples = 45  # up to t=4.4s, well before contact at t=6.0s
    ttc_series = []
    true_series = []

    for i in range(n_samples):
        t = i * dt
        distance = D0 - v * t
        width = k / distance
        tracker.update([make_detection(width, track_id=1)], t)
        ttc_series.append(tracker.min_ttc())
        true_series.append(distance / v)

    # Before MIN_SAMPLES_FOR_TTC samples exist, safety.py deliberately
    # withholds a TTC (see MIN_SAMPLES_FOR_TTC) -- a 2-3 point fit is the
    # unstable regime the noisy-flat test below shows spurious readings in.
    # That is intentional and is exactly why unresolved_vehicles() exists:
    # a caller must not read this early None as "not a threat".
    warm_up = MIN_SAMPLES_FOR_TTC - 1
    stable = ttc_series[warm_up:]
    assert all(x is not None for x in stable), "TTC should be defined once enough samples exist for a steadily approaching vehicle"

    # Monotonically non-increasing: TTC counts down as time passes.
    for earlier, later in zip(stable, stable[1:]):
        assert later <= earlier + 1e-9, "TTC must fall (or hold), never rise, for a steadily approaching vehicle"

    # Within a reasonable tolerance of the analytic remaining time.
    for computed, true_ttc in zip(ttc_series[warm_up:], true_series[warm_up:]):
        assert abs(computed - true_ttc) < 0.6, (
            f"computed TTC {computed:.2f} strayed too far from analytic {true_ttc:.2f}"
        )


def test_approaching_vehicle_ttc_matches_analytic_closely():
    # Pins down the fix for the lag that the old width-vs-time regression
    # had: fitting 1/width instead of width against time is exactly the
    # right model for constant-velocity approach (see module docstring), so
    # there should be near-zero systematic lag, not just "within 0.6s".
    # A future refactor that reintroduces a width-vs-time fit would fail
    # this test even though it might still pass the looser test above.
    # Same k choice as the previous test -- see its comment: a box that
    # starts too small sits below the noise-horizon safety margin and
    # correctly reports UNKNOWN rather than lag; that's not what this test
    # is checking, so keep the box comfortably above that regime.
    D0 = 60.0
    v = 10.0
    k = 4000.0
    dt = 0.1

    tracker = VehicleTracker()
    for i in range(45):
        t = i * dt
        distance = D0 - v * t
        width = k / distance
        tracker.update([make_detection(width, track_id=1)], t)
        if i < MIN_SAMPLES_FOR_TTC - 1:
            continue  # not enough samples yet -- TTC is withheld, see MIN_SAMPLES_FOR_TTC
        computed = tracker.min_ttc()
        true_ttc = distance / v
        assert computed is not None
        assert abs(computed - true_ttc) < 0.1, (
            f"at t={t:.2f}s computed TTC {computed:.3f} vs analytic {true_ttc:.3f} -- "
            "lag should be near zero for a true constant-velocity approach"
        )


def test_accelerating_vehicle_ttc_is_optimistic():
    # A vehicle that is speeding up as it approaches breaks the
    # constant-velocity assumption the 1/width-vs-time fit relies on. The
    # true 1/width trend is no longer a straight line here (it curves,
    # getting steeper as speed increases), so the linear fit under-reads the
    # true instantaneous rate and OVER-estimates TTC -- the same direction
    # of error the old width-vs-time formulation had everywhere. Phase 6
    # must not treat this number as exact; it should keep its own margin.
    D0 = 60.0
    v0 = 5.0  # m/s at t=0
    a = 4.0  # m/s^2 -- constant acceleration, speeding up
    k = 600.0
    dt = 0.1

    tracker = VehicleTracker()
    optimistic_count = 0
    checked = 0

    for i in range(40):
        t = i * dt
        instantaneous_speed = v0 + a * t
        distance = D0 - v0 * t - 0.5 * a * t * t
        if distance <= 0:
            break
        width = k / distance
        tracker.update([make_detection(width, track_id=1)], t)
        computed = tracker.min_ttc()
        if computed is None:
            continue
        # The instantaneous "if speed stayed exactly as it is right now"
        # TTC -- the best a memoryless, non-predictive estimator could hope
        # to match when the target is accelerating.
        true_instantaneous_ttc = distance / instantaneous_speed
        checked += 1
        if computed > true_instantaneous_ttc:
            optimistic_count += 1

    assert checked > 10
    # Document the direction plainly: for an accelerating approach the
    # estimate should be optimistic (reports more time than truly remains)
    # essentially every time, not just occasionally.
    assert optimistic_count / checked > 0.9, (
        "accelerating-vehicle TTC is expected to be optimistic almost always; "
        "if this changed, the direction of the bias needs re-documenting in safety.py"
    )


def test_shrinking_box_yields_no_ttc():
    # Vehicle receding: box gets smaller every frame. Must never produce a
    # TTC -- growth rate is negative, and a negative "time to contact" or an
    # inverted sign would be actively dangerous if fed to the FSM.
    tracker = VehicleTracker()
    widths = [80.0, 70.0, 60.0, 50.0, 40.0]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    readings = feed(tracker, widths, timestamps)
    assert all(r is None for r in readings)


def test_static_box_yields_no_ttc():
    # Parked vehicle: width doesn't change. Growth rate is ~0, must not be
    # divided into to produce a spurious near-instant or infinite TTC.
    tracker = VehicleTracker()
    widths = [50.0] * 5
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    readings = feed(tracker, widths, timestamps)
    assert all(r is None for r in readings)


def test_noisy_flat_widths_do_not_produce_spurious_ttc():
    # This is the false-alarm case the smoothing step exists to prevent: a
    # vehicle that is not actually approaching, but whose box measurement
    # jitters by a pixel or two frame to frame. Two adjacent raw samples
    # could easily show a large instantaneous "growth" by chance; the
    # least-squares fit over the whole window should see through it to the
    # true near-zero trend.
    tracker = VehicleTracker()
    widths = [50.0, 49.0, 51.0, 50.0, 49.5]  # noisy around a mean of ~49.9
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    readings = feed(tracker, widths, timestamps)
    assert readings[-1] is None, "noise-only fluctuation must not read as an approach"


def test_track_id_none_is_handled_without_crashing():
    tracker = VehicleTracker()
    det = make_detection(50.0, track_id=None)
    # Must not raise, and must not silently create a track keyed on None.
    tracker.update([det], 0.0)
    assert tracker.min_ttc() is None
    assert len(tracker.tracks) == 0


def test_track_dropped_after_15_missed_frames():
    tracker = VehicleTracker()
    tracker.update([make_detection(50.0, track_id=7)], 0.0)
    assert 7 in tracker.tracks

    # Feed frames with no detections at all for this track (something else,
    # or nothing, is in view). It should survive up to just under the
    # threshold and then be dropped.
    t = 0.1
    for i in range(MAX_MISSED_FRAMES - 1):
        tracker.update([], t)
        t += 0.1
    assert 7 in tracker.tracks, "must not drop before the missed-frame threshold"

    tracker.update([], t)  # this is the MAX_MISSED_FRAMES-th consecutive miss
    assert 7 not in tracker.tracks, "must drop once unseen for 15 frames"


def test_min_ttc_none_when_nothing_approaching():
    tracker = VehicleTracker()
    # one static, one receding -- neither is a threat
    for i, t in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
        tracker.update(
            [
                make_detection(50.0, track_id=1),
                make_detection(80.0 - i * 5, track_id=2),
            ],
            t,
        )
    assert tracker.min_ttc() is None


def test_min_ttc_is_minimum_across_several_approaching_vehicles():
    tracker = VehicleTracker()
    # Two vehicles approaching at different speeds -- min_ttc must report
    # the more urgent (smaller TTC) one.
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for t in timestamps:
        widths_fast = 40.0 + t * 60.0  # grows quickly -> small TTC
        widths_slow = 40.0 + t * 5.0  # grows slowly -> large TTC
        tracker.update(
            [
                make_detection(widths_fast, track_id=1),
                make_detection(widths_slow, track_id=2),
            ],
            t,
        )

    fast_ttc = tracker.get_ttc(1)
    slow_ttc = tracker.get_ttc(2)
    assert fast_ttc is not None and slow_ttc is not None
    assert fast_ttc < slow_ttc
    assert tracker.min_ttc() == pytest.approx(fast_ttc)


def test_person_class_is_never_a_vehicle_threat():
    # A person's box can grow just as fast as a vehicle's (someone walking
    # straight at the camera), but a person is not a collision threat for
    # the TTC/vehicle-approach logic -- the FSM handles pedestrians
    # separately. Confirm "person" is excluded from VEHICLE_CLASSES and
    # that feeding a fast-growing person produces no tracked vehicle at all.
    assert "person" not in VEHICLE_CLASSES

    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0, 75.0, 90.0]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=9, cls_name="person")], t)

    assert tracker.min_ttc() is None
    assert 9 not in tracker.tracks


def test_zero_elapsed_time_is_guarded():
    # Two detections stamped with the exact same timestamp (e.g. a bug
    # upstream, or duplicate frames) must not divide by a zero elapsed time.
    tracker = VehicleTracker()
    tracker.update([make_detection(50.0, track_id=1)], 1.0)
    tracker.update([make_detection(60.0, track_id=1)], 1.0)
    assert tracker.min_ttc() is None


def test_identical_consecutive_widths_guarded():
    tracker = VehicleTracker()
    tracker.update([make_detection(50.0, track_id=1)], 0.0)
    tracker.update([make_detection(50.0, track_id=1)], 0.1)
    assert tracker.min_ttc() is None


def test_single_observation_guarded():
    tracker = VehicleTracker()
    tracker.update([make_detection(50.0, track_id=1)], 0.0)
    assert tracker.min_ttc() is None
    assert tracker.get_ttc(1) is None


# ---------------------------------------------------------------------------
# unresolved_vehicles(): "we don't know yet" must be visible, not silent None.
#
# PLAN.md's Phase 6 rule only fires WAITING when min_ttc() is not None. A
# real approaching vehicle whose track is too new to have a TTC yet would
# otherwise fail to trigger WAITING and nothing would stop a false SAFE.
# unresolved_vehicles() exists so Phase 6 can refuse SAFE whenever a visible
# vehicle has no verdict at all, for any reason.
# ---------------------------------------------------------------------------

def test_new_track_with_too_few_samples_is_unresolved_not_dismissed():
    # A fast-approaching vehicle with only 3 samples -- one short of
    # MIN_SAMPLES_FOR_TTC. It must not yet report a TTC (too little data to
    # trust the fit), but it must also not be silently treated as "no
    # threat" -- it has to show up as unresolved so Phase 6 knows there is a
    # visible vehicle it cannot yet assess.
    assert MIN_SAMPLES_FOR_TTC == 4, "test assumes the documented default; update if this constant changes"

    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0]  # clearly approaching, but only 3 points
    timestamps = [0.0, 0.1, 0.2]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_ttc(1) is None, "3 samples is below MIN_SAMPLES_FOR_TTC, TTC must be withheld"
    assert tracker.unresolved_vehicles() == 1, "a visible vehicle with no TTC yet must count as unresolved"


def test_track_resolves_once_enough_samples_arrive():
    # Continue the same approaching vehicle from the previous test past the
    # MIN_SAMPLES_FOR_TTC threshold: it should get a real TTC and drop out
    # of the unresolved count.
    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0, 75.0, 90.0]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    for i, (width, t) in enumerate(zip(widths, timestamps)):
        tracker.update([make_detection(width, track_id=1)], t)
        if i < MIN_SAMPLES_FOR_TTC - 1:
            assert tracker.unresolved_vehicles() == 1
        else:
            assert tracker.get_ttc(1) is not None
            assert tracker.unresolved_vehicles() == 0


def test_visible_receding_vehicle_with_full_window_is_not_approaching_not_unresolved():
    # Supersedes the earlier (deliberately conservative) version of this
    # test. A receding vehicle assessed over a FULL window is a real
    # assessment -- NOT_APPROACHING -- and must not block SAFE by counting
    # as unresolved. Only "not enough data yet" (UNKNOWN) should do that;
    # see test_parked_vehicle_settles_to_not_approaching_and_is_resolved for
    # why this distinction has to exist at all.
    tracker = VehicleTracker()
    widths = [80.0, 70.0, 60.0, 50.0, 40.0]  # clearly receding
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_ttc(1) is None
    assert tracker.get_status(1) == NOT_APPROACHING
    assert tracker.unresolved_vehicles() == 0


def test_parked_vehicle_settles_to_not_approaching_and_is_resolved():
    # THE real-world scenario this whole distinction exists for: at an
    # actual Indian crossing there is almost always a parked vehicle, a
    # stopped bus, or a stationary rickshaw somewhere in frame. If a merely
    # parked vehicle counted as "unresolved" forever, unresolved_vehicles()
    # would never reach zero and the system could never say SAFE, which is
    # its own safety failure (a system stuck permanently on "wait" teaches
    # the user to ignore it). A parked vehicle, once fully assessed, must
    # resolve to NOT_APPROACHING and stop blocking SAFE.
    tracker = VehicleTracker()
    widths = [50.0] * 6  # parked: perfectly static, well past a full window
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_status(1) == NOT_APPROACHING
    assert tracker.get_ttc(1) is None
    assert tracker.unresolved_vehicles() == 0


def test_new_track_with_flat_looking_widths_is_still_unknown_not_assessed():
    # A brand new track with only 3 samples that happen to look flat must
    # still be UNKNOWN, not NOT_APPROACHING -- insufficient data must never
    # masquerade as an assessment, even when the few samples we do have look
    # like they'd support one. This is the ordering rule from the module
    # docstring: sample count is checked before the growth rate.
    tracker = VehicleTracker()
    widths = [50.0, 50.0, 50.0]  # looks parked, but too few samples to say so
    timestamps = [0.0, 0.1, 0.2]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_status(1) == UNKNOWN
    assert tracker.get_ttc(1) is None
    assert tracker.unresolved_vehicles() == 1


def test_approaching_vehicle_with_full_window_has_approaching_status():
    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0, 75.0, 90.0]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_status(1) == APPROACHING
    assert tracker.get_ttc(1) is not None
    assert tracker.unresolved_vehicles() == 0


def test_bad_box_with_full_window_is_unknown_not_not_approaching():
    # A non-positive width is a bad detection, not evidence the vehicle is
    # safe -- it must land on UNKNOWN, never on NOT_APPROACHING, even though
    # the window is otherwise "full" in sample count.
    tracker = VehicleTracker()
    widths = [50.0, 50.0, 0.0, 50.0]  # a zero-width glitch frame in the window
    timestamps = [0.0, 0.1, 0.2, 0.3]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_status(1) == UNKNOWN
    assert tracker.get_ttc(1) is None
    assert tracker.unresolved_vehicles() == 1


def test_noisy_flat_status_settles_without_oscillating():
    # The noisy-flat static box (the classic false-alarm case) should
    # settle to NOT_APPROACHING once the window fills, and -- this is the
    # important part for Phase 6's hysteresis -- it should not then flicker
    # back and forth between UNKNOWN/NOT_APPROACHING/APPROACHING on
    # subsequent frames as the noisy window slides.
    #
    # HISTORY: this test originally failed. With MIN_APPROACH_RATE at its
    # earlier value of 1e-4 (tuned only against a single 5-sample noisy
    # window), this longer 12-frame noisy sequence measurably oscillated:
    # UNKNOWN, UNKNOWN, UNKNOWN, APPROACHING, NOT_APPROACHING, APPROACHING,
    # NOT_APPROACHING, APPROACHING, APPROACHING, NOT_APPROACHING,
    # NOT_APPROACHING, NOT_APPROACHING -- because plausible pixel jitter
    # (+/-1-2px on a ~50px box) produced apparent approach rates up to
    # ~1.2e-3, over 10x the old threshold. MIN_APPROACH_RATE was raised to
    # 2e-3 to clear that measured noise ceiling -- see its definition in
    # safety.py for the tradeoff that creates (reduced sensitivity to very
    # slow, distant approaches).
    tracker = VehicleTracker()
    # 12 frames of noise around a mean of ~50px -- enough to slide the
    # 5-sample window several times over.
    noisy_widths = [
        50.0, 49.0, 51.0, 50.0, 49.5,
        50.5, 49.2, 50.8, 49.7, 50.3,
        49.4, 50.6,
    ]
    timestamps = [i * 0.1 for i in range(len(noisy_widths))]

    statuses = []
    for width, t in zip(noisy_widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)
        statuses.append(tracker.get_status(1))

    # First MIN_SAMPLES_FOR_TTC - 1 readings are UNKNOWN (not enough data).
    warm_up = MIN_SAMPLES_FOR_TTC - 1
    assert all(s == UNKNOWN for s in statuses[:warm_up])

    # From there on, a genuinely flat/noisy signal must settle to
    # NOT_APPROACHING and stay there -- no oscillation for Phase 6's
    # hysteresis to have to absorb.
    settled = statuses[warm_up:]
    assert all(s == NOT_APPROACHING for s in settled), (
        f"expected the noisy-flat box to settle to {NOT_APPROACHING} and stay there, "
        f"got status sequence {statuses}"
    )


def test_unresolved_vehicles_ignores_tracks_that_left_the_frame():
    # A track that briefly isn't in this frame's detections (frames_since_seen
    # > 0) but hasn't been dropped yet is not "a vehicle we can currently see
    # and cannot assess" -- it isn't visible right now at all, so it must not
    # inflate the unresolved count.
    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0]  # 3 samples: unresolved while visible
    timestamps = [0.0, 0.1, 0.2]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)
    assert tracker.unresolved_vehicles() == 1

    # Now it drops out of view for a few frames, well under MAX_MISSED_FRAMES.
    tracker.update([], 0.3)
    tracker.update([], 0.4)
    assert 1 in tracker.tracks, "not yet aged out"
    assert tracker.unresolved_vehicles() == 0, "not currently visible, so not counted as unresolved"


def test_person_never_appears_in_unresolved_vehicles():
    # A person is never a vehicle track at all (see
    # test_person_class_is_never_a_vehicle_threat), so a fast-approaching
    # person must not inflate the unresolved count either.
    tracker = VehicleTracker()
    widths = [30.0, 45.0, 60.0]
    timestamps = [0.0, 0.1, 0.2]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=9, cls_name="person")], t)
    assert tracker.unresolved_vehicles() == 0


def test_noisy_flat_static_box_still_none_with_min_samples_requirement():
    # Re-confirm the false-alarm guard still holds now that a minimum
    # sample count is also enforced -- the two guards are independent and
    # both need to hold.
    tracker = VehicleTracker()
    widths = [50.0, 49.0, 51.0, 50.0, 49.5]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    readings = feed(tracker, widths, timestamps)
    assert readings[-1] is None, "noise-only fluctuation must not read as an approach"


# ---------------------------------------------------------------------------
# The scale-aware noise horizon (see the module docstring's derivation):
# a single fixed threshold on approach_rate cannot work because the noise a
# fixed pixel jitter injects into 1/width scales with 1/width^2, so a small
# distant box looks noisier than a large close one for the same jitter.
# Instead safety.py converts the assumed jitter into a TTC horizon
# (w * window_duration / JITTER_PIXELS) and compares the computed TTC
# against that, rather than comparing the raw rate against a constant.
# ---------------------------------------------------------------------------

def test_noisy_flat_12_frame_sequence_settles_without_oscillating_v2():
    # This is the sequence that exposed the previous (fixed-threshold)
    # scheme's failure: with MIN_APPROACH_RATE alone at 1e-4 it oscillated
    # between APPROACHING and NOT_APPROACHING frame to frame as the 5-sample
    # window slid across different noise combinations. Re-run under the
    # noise-horizon scheme and report the full status sequence.
    tracker = VehicleTracker()
    noisy_widths = [
        50.0, 49.0, 51.0, 50.0, 49.5,
        50.5, 49.2, 50.8, 49.7, 50.3,
        49.4, 50.6,
    ]
    timestamps = [i * 0.1 for i in range(len(noisy_widths))]

    statuses = []
    for width, t in zip(noisy_widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)
        statuses.append(tracker.get_status(1))

    warm_up = MIN_SAMPLES_FOR_TTC - 1
    assert all(s == UNKNOWN for s in statuses[:warm_up])

    settled = statuses[warm_up:]
    assert all(s == NOT_APPROACHING for s in settled), (
        f"expected the noisy-flat box to settle to {NOT_APPROACHING} and stay there under the "
        f"noise-horizon scheme, got status sequence {statuses}"
    )


def test_slow_distant_approach_becomes_approaching_well_before_5s_ttc():
    # THE case the old fixed MIN_APPROACH_RATE got wrong: a slowly-closing,
    # fairly distant vehicle (v=2 m/s, starting 15m out -> true contact at
    # 7.5s) has a small raw approach_rate, similar in size to what pixel
    # jitter alone could fake. Under a fixed-rate threshold tuned to reject
    # that jitter (as the previous iteration of this module used), this
    # vehicle could be misread as NOT_APPROACHING -- read as SAFETY EVIDENCE
    # for a vehicle that is, in fact, still closing. The noise horizon must
    # not make that mistake: it should flip to APPROACHING with a real TTC
    # well before the true TTC nears PLAN.md's 5.0s decision threshold.
    D0 = 15.0
    v = 2.0  # m/s -- a slow, crawling approach
    k = 750.0  # width ~50px at D0, comfortably above the noise-horizon floor
    dt = 0.1

    tracker = VehicleTracker()
    first_approaching_true_ttc = None

    for i in range(80):
        t = i * dt
        distance = D0 - v * t
        if distance <= 0:
            break
        width = k / distance
        tracker.update([make_detection(width, track_id=1)], t)
        status = tracker.get_status(1)
        true_ttc = distance / v

        if status == APPROACHING and first_approaching_true_ttc is None:
            first_approaching_true_ttc = true_ttc
            computed_ttc = tracker.get_ttc(1)
            assert computed_ttc is not None
            assert abs(computed_ttc - true_ttc) < 0.5

    assert first_approaching_true_ttc is not None, "slow approach should eventually resolve to APPROACHING"
    assert first_approaching_true_ttc > DECISION_TTC_THRESHOLD, (
        f"expected APPROACHING to be reached while true TTC ({first_approaching_true_ttc:.2f}s) "
        f"is still well above the {DECISION_TTC_THRESHOLD}s decision threshold, "
        "not only once it's already urgent"
    )


def test_small_box_with_horizon_below_margin_is_unknown_not_not_approaching():
    # A box small enough (and/or a window short enough) that the noise
    # horizon itself falls below HORIZON_SAFETY_MARGIN * DECISION_TTC_THRESHOLD
    # must resolve to UNKNOWN, not NOT_APPROACHING, when its TTC exceeds that
    # horizon -- we cannot honestly rule out a real approach at that scale.
    # A tiny, slowly-growing box (5px, growing by 0.05px/frame) is exactly
    # this case: horizon = w*T/delta ~= 5*0.3/1.5 = 1.0s, far below the 10s
    # (2x margin over the 5s decision threshold) needed to trust a
    # NOT_APPROACHING call.
    tracker = VehicleTracker()
    widths = [5.0, 5.05, 5.1, 5.15]
    timestamps = [0.0, 0.1, 0.2, 0.3]
    for width, t in zip(widths, timestamps):
        tracker.update([make_detection(width, track_id=1)], t)

    assert tracker.get_status(1) == UNKNOWN
    assert tracker.get_ttc(1) is None
    assert tracker.unresolved_vehicles() == 1
