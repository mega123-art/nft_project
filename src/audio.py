"""
Phase 7: turn a Phase 6 Verdict into speech + earcons without ever blocking
the capture loop.

PLAN.md's own words for this file: "pyttsx3 for speech, running on a
separate thread or the video will stutter. Use a queue with maxsize=1 and
drop stale messages. pygame.mixer for earcons: a rising two-tone for safe, a
repeated low pulse for danger. Earcons are much faster than speech and
matter more for urgency. Rate limit: do not repeat the same announcement
within 2 seconds, and do not announce anything more than once per 2 seconds
overall. Speak only on state change, not every frame."

This module does not invent a second set of wordings for the states -- the
reason strings the user should hear already live in fsm.py's Verdict.reason
(e.g. "wait, vehicle approaching", "safe to cross now"), written by Phase 6
specifically to be explainable in the viva. Re-wording them here would let
the spoken text and the on-screen/logged text drift apart for no benefit, so
`announce()` speaks `verdict.reason` (plus, when known, the crosswalk
direction) rather than a Phase 7-only phrase table.

THREE DESIGN DECISIONS THAT AREN'T IN PLAN.md's ONE PARAGRAPH, and why:

1. URGENCY ORDERING (earcons must not queue up behind speech). PLAN.md
   itself says earcons "matter more for urgency" than speech, which only
   makes sense if a danger earcon can actually cut ahead of whatever speech
   is in flight. pygame.mixer plays earcons on their own Channel, entirely
   separate from the pyttsx3 worker thread and its queue -- there is no
   shared lock, no shared queue slot, and no "wait for speech to finish"
   step anywhere in the earcon path. On top of that, WAITING's danger pulse
   calls Channel.stop() on itself before re-playing, and a WAITING
   transition also drops (does not merely queue) any pending SAFE-flavoured
   speech that hasn't started yet (see _speak_worker). A late warning is
   worse than no warning, because by the time it lands the user may already
   be off the kerb, so the danger path is built to always win a race against
   speech.

2. THE GLOBAL RATE LIMIT MUST NOT SUPPRESS A NEW WARNING. A blanket "nothing
   more than once per 2 seconds" is exactly what PLAN.md asks for in its one
   sentence, but taken completely literally it would silently eat a
   SAFE -> WAITING transition that lands 1.5s after some earlier, unrelated
   announcement -- precisely the transition that must never be swallowed
   (see fsm.py's own module docstring: a false SAFE is the one unacceptable
   failure). fsm.py already treats "entering a cautious state" and "leaving
   one" asymmetrically (instant to withdraw SAFE_TO_CROSS, slow to grant
   it -- see CrossingFSM._apply_hysteresis's docstring); this module applies
   the same asymmetry to sound: a transition INTO WAITING is exempt from
   both rate limits (global and per-message) and is always announced
   immediately. Only "routine" announcements (SEARCHING, and re-confirmations
   of SAFE_TO_CROSS/WAITING) respect the 2-second limits. This is a
   deliberate, documented departure from PLAN.md's literal wording, made
   because the literal wording conflicts with PLAN.md's own overriding
   safety rule.

3. GRACEFUL DEGRADATION + CLEAN SHUTDOWN. No espeak-ng, no sound card, a
   teammate's machine with neither -- none of that should be able to kill a
   vision system that otherwise works. Both engines are constructed inside
   try/except at __init__ time; a failure is logged exactly once and the
   announcer quietly becomes a no-op for that channel (speech, earcons, or
   both) rather than raising out of announce()/close(). The pyttsx3 worker
   runs as a daemon thread so a forgotten close() cannot hang process exit;
   close() itself pushes a sentinel and joins with a timeout so it never
   blocks indefinitely either.
"""

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard requirement elsewhere
    np = None

from fsm import SAFE_TO_CROSS, WAITING

# --- Rate limiting --------------------------------------------------------
# See point 2 above: these apply to routine announcements only. A fresh
# transition into WAITING always bypasses both.
SAME_MESSAGE_COOLDOWN_S = 2.0
GLOBAL_COOLDOWN_S = 2.0

# --- Earcon tone parameters ------------------------------------------------
SAMPLE_RATE = 22050
SAFE_TONES_HZ = (523, 784)     # rising two-tone (C5 -> G5) -- "go"
DANGER_TONE_HZ = 220           # low pulse, repeated -- "stop/wait"
DANGER_PULSES = 3

_SENTINEL = object()  # pushed onto the speech queue to ask the worker to exit


def _tone(freq_hz, duration_s, sample_rate=SAMPLE_RATE, volume=0.5):
    """One sine-wave tone as int16 mono samples, faded in/out a few ms so it
    doesn't click. Generated programmatically -- no audio files to ship, and
    it's a one-line thing to explain in the viva."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq_hz * t)

    fade_n = max(1, int(sample_rate * 0.01))
    fade_n = min(fade_n, n // 2) if n >= 2 else 0
    if fade_n > 0:
        envelope = np.ones(n)
        envelope[:fade_n] = np.linspace(0, 1, fade_n)
        envelope[-fade_n:] = np.linspace(1, 0, fade_n)
        wave = wave * envelope

    samples = (wave * volume * 32767).astype(np.int16)
    return samples


def _build_safe_earcon_samples():
    """Rising two-tone: two short notes back to back, second higher than
    first."""
    a = _tone(SAFE_TONES_HZ[0], 0.12)
    b = _tone(SAFE_TONES_HZ[1], 0.16)
    gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.int16)
    return np.concatenate([a, gap, b])


def _build_danger_earcon_samples():
    """Repeated low pulse -- three short low beeps, not one long tone, so it
    reads as "alert" rather than a single flat note."""
    pulse = _tone(DANGER_TONE_HZ, 0.12)
    gap = np.zeros(int(SAMPLE_RATE * 0.08), dtype=np.int16)
    parts = []
    for i in range(DANGER_PULSES):
        parts.append(pulse)
        if i != DANGER_PULSES - 1:
            parts.append(gap)
    return np.concatenate(parts)


class AudioAnnouncer:
    """Speaks Verdict.reason on state change (threaded TTS) and plays an
    earcon alongside it. Never raises out of announce()/close(), and never
    blocks the caller for longer than a queue.put_nowait() takes.

    tts_engine_factory / mixer_factory / sound_factory are injectable so
    tests can run this whole class's logic without a sound card -- see
    tests/test_audio.py. Left at their defaults, they are pyttsx3.init(),
    pygame.mixer (init + Channel + Sound).
    """

    def __init__(self, tts_engine_factory=None, mixer_factory=None, now=time.monotonic):
        self._now = now
        self._last_state = None
        self._last_message = None
        # -inf, not 0.0: a fake clock in tests (or a monotonic clock that
        # happens to start near 0) must never look like "just announced".
        self._last_message_time = float("-inf")
        self._last_announce_time = float("-inf")

        self._speech_queue = queue.Queue(maxsize=1)
        self._engine = self._init_tts(tts_engine_factory)
        self._thread = None
        if self._engine is not None:
            self._thread = threading.Thread(target=self._speak_worker, daemon=True)
            self._thread.start()

        self._safe_sound, self._danger_sound, self._channel = self._init_earcons(mixer_factory)

    # -- init helpers, each isolated so one failing engine doesn't take the
    # other down with it -----------------------------------------------
    def _init_tts(self, tts_engine_factory):
        try:
            if tts_engine_factory is not None:
                return tts_engine_factory()
            import pyttsx3
            return pyttsx3.init()
        except Exception:
            # Logged once, here, then this announcer runs silent for speech
            # for its whole life -- see module docstring point 3.
            logger.warning("AudioAnnouncer: TTS init failed, running without speech", exc_info=True)
            return None

    def _init_earcons(self, mixer_factory):
        try:
            if mixer_factory is not None:
                mixer = mixer_factory()
            else:
                import pygame

                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
                mixer = pygame.mixer
            safe_sound = mixer.Sound(buffer=_build_safe_earcon_samples().tobytes())
            danger_sound = mixer.Sound(buffer=_build_danger_earcon_samples().tobytes())
            channel = mixer.Channel(0)
            return safe_sound, danger_sound, channel
        except Exception:
            logger.warning("AudioAnnouncer: mixer init failed, running without earcons", exc_info=True)
            return None, None, None

    # -- speech worker -----------------------------------------------------
    def _speak_worker(self):
        while True:
            message = self._speech_queue.get()
            if message is _SENTINEL:
                return
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception:
                # A mid-life TTS failure (e.g. the audio device disappears)
                # must not kill the worker thread or the process -- log once
                # per occurrence and keep pulling from the queue.
                logger.warning("AudioAnnouncer: speech playback failed", exc_info=True)

    def _enqueue_speech(self, message, drop_pending=False):
        """Put `message` on the maxsize=1 queue, never blocking the caller.

        If the queue is already full, the stale message sitting in it is
        dropped in favour of the new one -- by the time an old announcement
        would play, the world has moved on (see module + PLAN.md). When
        drop_pending is True (used for an urgent WAITING announcement, see
        point 1) this also happens even though the queue's own maxsize=1
        already achieves the same "keep only the newest" effect -- it is
        kept explicit here so the intent reads clearly at the call site.
        """
        if self._engine is None:
            return
        while True:
            try:
                self._speech_queue.put_nowait(message)
                return
            except queue.Full:
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    pass
                # loop and retry the put; at most one stale item to clear

    # -- earcons -------------------------------------------------------
    def _play_earcon(self, state):
        if self._channel is None:
            return
        try:
            if state == SAFE_TO_CROSS:
                self._channel.play(self._safe_sound)
            elif state == WAITING:
                # stop() first: a danger pulse must never be stuck queued
                # behind an earlier one still playing (point 1) -- restart
                # it fresh so the newest warning is always heard promptly.
                self._channel.stop()
                self._channel.play(self._danger_sound)
            # SEARCHING gets no earcon -- "still looking" is not urgent and
            # PLAN.md doesn't ask for a sound for it.
        except Exception:
            logger.warning("AudioAnnouncer: earcon playback failed", exc_info=True)

    # -- public API ----------------------------------------------------
    def announce(self, verdict):
        """Speak + play an earcon for `verdict`, if it's worth announcing.

        Only ever acts on a state CHANGE (verdict.changed, mirroring
        fsm.py's own "changed" flag -- Phase 6 already did the hysteresis
        work of deciding what counts as a real change, so this module
        trusts it rather than re-deriving change detection from raw
        states). Never raises.
        """
        try:
            self._announce(verdict)
        except Exception:
            # Audio must never be able to take the capture loop down.
            logger.warning("AudioAnnouncer: announce() failed", exc_info=True)

    def _announce(self, verdict):
        if not verdict.changed:
            return

        state = verdict.state
        message = verdict.reason
        if state == SAFE_TO_CROSS and verdict.crosswalk_direction not in (None, "center"):
            message = f"{message}, crossing is to your {verdict.crosswalk_direction}"

        now = self._now()
        is_new_warning = state == WAITING and self._last_state != WAITING

        if not is_new_warning:
            # Routine announcement: subject to both rate limits (point 2).
            same_message_recent = (
                message == self._last_message
                and (now - self._last_message_time) < SAME_MESSAGE_COOLDOWN_S
            )
            any_recent = (now - self._last_announce_time) < GLOBAL_COOLDOWN_S
            if same_message_recent or any_recent:
                self._last_state = state
                return

        # Either this is a fresh WAITING warning (always gets through,
        # point 2) or it cleared both rate limits.
        self._enqueue_speech(message, drop_pending=is_new_warning)
        self._play_earcon(state)

        self._last_state = state
        self._last_message = message
        self._last_message_time = now
        self._last_announce_time = now

    def close(self, timeout=2.0):
        """Stop the speech thread promptly. Safe to call more than once."""
        if self._thread is not None and self._thread.is_alive():
            self._enqueue_speech_sentinel()
            self._thread.join(timeout=timeout)
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:
            pass

    def _enqueue_speech_sentinel(self):
        # Bypass the "drop stale, keep newest real message" logic above --
        # the sentinel must always get through so the thread actually exits.
        while True:
            try:
                self._speech_queue.put_nowait(_SENTINEL)
                return
            except queue.Full:
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    pass
