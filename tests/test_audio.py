"""
Phase 7 tests for src/audio.py.

Like test_fsm.py and test_safety.py, these are synthetic: no sound card is
needed or used. We inject fake TTS-engine and mixer factories so the whole
threading / rate-limiting / urgency-ordering logic can be exercised in CI.
Each test encodes one specific rule from PLAN.md's Phase 7 section or one of
the extra design decisions documented in audio.py's module docstring, not
just "whatever the code currently does".
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio import AudioAnnouncer, SAME_MESSAGE_COOLDOWN_S, GLOBAL_COOLDOWN_S
from fsm import Verdict, SEARCHING, WAITING, SAFE_TO_CROSS


def make_verdict(state, reason=None, changed=True, direction=None):
    if reason is None:
        reason = {
            SEARCHING: "looking for a crossing",
            WAITING: "wait, vehicle approaching",
            SAFE_TO_CROSS: "safe to cross now",
        }[state]
    return Verdict(
        state=state,
        reason=reason,
        raw_state=state,
        raw_reason=reason,
        min_ttc=None,
        crosswalk_offset=None,
        crosswalk_direction=direction,
        changed=changed,
    )


class FakeEngine:
    """Stand-in for pyttsx3.Engine: records what it was told to say."""

    def __init__(self, fail=False, hang_event=None):
        self.fail = fail
        self.hang_event = hang_event
        self.said = []
        self.stopped = False

    def say(self, message):
        if self.fail:
            raise RuntimeError("synthetic TTS failure")
        if self.hang_event is not None:
            self.hang_event.wait(timeout=5)
        self.said.append(message)

    def runAndWait(self):
        pass

    def stop(self):
        self.stopped = True


class FailingFactory:
    def __call__(self):
        raise RuntimeError("no espeak-ng on this machine")


class FakeSound:
    def __init__(self, buffer=None):
        self.buffer = buffer


class FakeChannel:
    def __init__(self):
        self.played = []
        self.stop_calls = 0
        self.busy = False

    def play(self, sound):
        self.played.append(sound)
        self.busy = True

    def stop(self):
        self.stop_calls += 1
        self.busy = False

    def get_busy(self):
        return self.busy


class FakeMixer:
    """Stand-in for pygame.mixer: one shared channel, like real
    mixer.Channel(0) would be if we always ask for channel 0."""

    def __init__(self, fail=False):
        self.fail = fail
        self._channel = FakeChannel()

    def Sound(self, buffer=None):
        if self.fail:
            raise RuntimeError("no audio device")
        return FakeSound(buffer)

    def Channel(self, index):
        if self.fail:
            raise RuntimeError("no audio device")
        return self._channel


def make_announcer(engine=None, mixer=None, now=None):
    engine = FakeEngine() if engine is None else engine
    mixer = FakeMixer() if mixer is None else mixer
    kwargs = {}
    if now is not None:
        kwargs["now"] = now
    return AudioAnnouncer(
        tts_engine_factory=lambda: engine,
        mixer_factory=lambda: mixer,
        **kwargs,
    ), engine, mixer


def wait_for(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --- state change / silence -----------------------------------------------

def test_speaks_on_state_change():
    announcer, engine, mixer = make_announcer()
    announcer.announce(make_verdict(WAITING, changed=True))
    assert wait_for(lambda: engine.said)
    announcer.close()


def test_silent_when_state_unchanged_across_many_frames():
    announcer, engine, mixer = make_announcer()
    announcer.announce(make_verdict(WAITING, changed=True))
    assert wait_for(lambda: engine.said)
    engine.said.clear()

    for _ in range(50):
        announcer.announce(make_verdict(WAITING, changed=False))
    time.sleep(0.1)
    assert engine.said == []
    announcer.close()


# --- rate limiting -----------------------------------------------------

def test_same_announcement_suppressed_within_two_seconds_then_allowed():
    clock = [0.0]
    announcer, engine, mixer = make_announcer(now=lambda: clock[0])

    announcer.announce(make_verdict(SEARCHING, changed=True))
    assert wait_for(lambda: engine.said)
    assert len(engine.said) == 1

    # Same message, well inside the cooldown -> suppressed.
    clock[0] += SAME_MESSAGE_COOLDOWN_S - 0.5
    announcer.announce(make_verdict(SEARCHING, changed=True))
    time.sleep(0.05)
    assert len(engine.said) == 1

    # Same message, cooldown elapsed -> allowed again.
    clock[0] += SAME_MESSAGE_COOLDOWN_S + 0.1
    announcer.announce(make_verdict(SEARCHING, changed=True))
    assert wait_for(lambda: len(engine.said) == 2)
    announcer.close()


def test_global_rate_limit_holds_for_routine_announcements():
    clock = [0.0]
    announcer, engine, mixer = make_announcer(now=lambda: clock[0])

    announcer.announce(make_verdict(SEARCHING, reason="looking for a crossing", changed=True))
    assert wait_for(lambda: engine.said)

    # A different, but still routine, message inside the global cooldown
    # must also be held back -- "more than once per 2s overall".
    clock[0] += 0.5
    announcer.announce(
        make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True)
    )
    time.sleep(0.05)
    assert len(engine.said) == 1

    clock[0] += GLOBAL_COOLDOWN_S + 0.1
    announcer.announce(
        make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True)
    )
    assert wait_for(lambda: len(engine.said) == 2)
    announcer.close()


def test_safe_to_waiting_transition_is_not_suppressed_by_rate_limit():
    """Design point 2: entering WAITING must get through even if some
    unrelated announcement just fired inside the global cooldown window."""
    clock = [0.0]
    announcer, engine, mixer = make_announcer(now=lambda: clock[0])

    announcer.announce(make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True))
    assert wait_for(lambda: engine.said)

    # Only 0.2s later -- well inside GLOBAL_COOLDOWN_S -- a vehicle appears
    # and the FSM flips to WAITING. This must NOT be swallowed.
    clock[0] += 0.2
    announcer.announce(make_verdict(WAITING, reason="wait, vehicle approaching", changed=True))
    assert wait_for(lambda: len(engine.said) == 2)
    assert engine.said[-1] == "wait, vehicle approaching"
    announcer.close()


# --- queue behaviour -----------------------------------------------------

def test_queue_maxsize_one_drops_stale_message_without_blocking_caller():
    """Exercises the queue mechanism directly (._enqueue_speech), which is
    what announce() calls once it has decided a message is worth sending --
    the rate limiter's own job is tested separately above. maxsize=1 means a
    second put while the worker is still busy with the first must replace
    whatever's waiting, not queue up, and must never block the caller."""
    hang_event = __import__("threading").Event()
    engine = FakeEngine(hang_event=hang_event)
    announcer, engine, mixer = make_announcer(engine=engine)

    # First message: picked up by the worker immediately and hangs inside
    # say() until we release hang_event, simulating slow speech playback.
    announcer._enqueue_speech("first")
    # Give the worker a moment to actually pull it off the queue so the
    # queue itself is empty and ready to accept the next put.
    time.sleep(0.1)

    # Two more messages while the worker is still busy with "first". Neither
    # call should block the caller (maxsize=1 -> put_nowait + drop-and-retry).
    start = time.monotonic()
    announcer._enqueue_speech("second")
    announcer._enqueue_speech("third")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0  # did not block waiting on the busy worker

    hang_event.set()
    assert wait_for(lambda: len(engine.said) >= 2)
    # "second" was dropped in favour of "third" (only one slot to hold a
    # pending message) -- "first" (already in flight) still played.
    assert engine.said[0] == "first"
    assert "third" in engine.said
    assert "second" not in engine.said
    announcer.close()


# --- graceful degradation -----------------------------------------------

def test_failing_tts_init_leaves_working_silent_announcer():
    mixer = FakeMixer()
    announcer = AudioAnnouncer(tts_engine_factory=FailingFactory(), mixer_factory=lambda: mixer)
    # Must not raise, and announce() must still be safely callable.
    announcer.announce(make_verdict(WAITING, changed=True))
    time.sleep(0.05)
    announcer.close()


def test_failing_mixer_init_leaves_working_silent_announcer():
    engine = FakeEngine()
    announcer = AudioAnnouncer(
        tts_engine_factory=lambda: engine,
        mixer_factory=FailingFactory(),
    )
    announcer.announce(make_verdict(WAITING, changed=True))
    assert wait_for(lambda: engine.said)  # speech still works
    announcer.close()


def test_both_failing_never_raises():
    announcer = AudioAnnouncer(
        tts_engine_factory=FailingFactory(),
        mixer_factory=FailingFactory(),
    )
    announcer.announce(make_verdict(WAITING, changed=True))
    announcer.announce(make_verdict(SAFE_TO_CROSS, changed=True))
    announcer.close()


# --- shutdown ------------------------------------------------------------

def test_close_terminates_thread_promptly():
    announcer, engine, mixer = make_announcer()
    announcer.announce(make_verdict(WAITING, changed=True))
    assert wait_for(lambda: engine.said)
    thread = announcer._thread
    announcer.close(timeout=2.0)
    assert wait_for(lambda: not thread.is_alive(), timeout=1.0)


# --- urgency ordering ------------------------------------------------------

def test_danger_earcon_not_blocked_behind_in_flight_speech():
    """Design point 1: earcons live on their own mixer channel, untouched by
    whatever the speech thread is doing, so a WAITING earcon plays even
    while a long-running speech utterance is still in progress."""
    hang_event = __import__("threading").Event()
    engine = FakeEngine(hang_event=hang_event)
    announcer, engine, mixer = make_announcer(engine=engine)

    announcer.announce(make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True))
    time.sleep(0.1)  # worker now blocked inside say(), simulating slow speech

    start = time.monotonic()
    announcer.announce(make_verdict(WAITING, reason="wait, vehicle approaching", changed=True))
    elapsed = time.monotonic() - start

    # The earcon call itself must return promptly, and the danger sound must
    # have been handed to the channel immediately -- it does not wait on the
    # (still-hanging) speech worker at all.
    assert elapsed < 0.5
    assert mixer._channel.played, "danger earcon should have played without waiting on speech"

    hang_event.set()
    announcer.close()


def test_waiting_earcon_restarts_channel_rather_than_queueing():
    """A second WAITING earcon while the channel is still 'busy' must stop
    and replay rather than being left to queue behind the first."""
    announcer, engine, mixer = make_announcer()
    announcer.announce(make_verdict(WAITING, reason="first warning", changed=True))
    time.sleep(0.05)
    assert mixer._channel.stop_calls >= 1  # stop() called before every WAITING play
    plays_before = len(mixer._channel.played)

    # Force back to non-WAITING then re-enter WAITING to simulate a fresh
    # warning arriving.
    announcer.announce(make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True))
    announcer.announce(make_verdict(WAITING, reason="second warning", changed=True))
    time.sleep(0.05)
    assert len(mixer._channel.played) > plays_before
    announcer.close()


def test_crosswalk_direction_spoken_on_safe_to_cross():
    announcer, engine, mixer = make_announcer()
    announcer.announce(
        make_verdict(SAFE_TO_CROSS, reason="safe to cross now", changed=True, direction="right")
    )
    assert wait_for(lambda: engine.said)
    assert "right" in engine.said[0]
    announcer.close()
