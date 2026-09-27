# Labelling conventions — PROPOSAL, not yet agreed

**Status: DRAFT.** PLAN.md (Phase 4, step 6) says these conventions must be
agreed by the whole team *before* anyone opens Roboflow, and names
inconsistency between labellers as the main cause of poor accuracy in
projects like this one. Read this together, argue about it, change what you
disagree with, then have every labeller sign off at the bottom. Nobody
starts labelling frames against a rule they have not read and agreed to.

This document currently reflects one person's proposal. Treat every rule
below as "unless we decide otherwise", not as settled.

---

## 1. Box edges on a traffic signal head

**Proposal: box the housing around the lit lamp — the dark casing of that
one lamp's cell. Not the bare lens, not the hood/visor, and not the whole
multi-lamp head.**

Concretely, on a vertical 3-lamp signal with the green lamp lit, you draw
one box around the green lamp's casing cell: roughly the bottom third of
the head. You do not box all three lamps, and you do not box only the
glowing circle.

Reasoning:
- The lit lens alone is a tiny, low-contrast target (often under 10px) and
  its apparent edge changes with bloom/glare, which makes boxes inconsistent
  between labellers and between frames of the same signal.
- The housing has a hard, stable physical edge that is easy to see even when
  the lens is small or slightly overexposed, so two labellers converge on
  the same box.
- The hood/visor (the sun shade above each lens) should be **excluded** when
  it is a separate protruding piece, because including it inflates the box
  with background sky/wall and lowers IoU against a lens-tight ground truth
  used to describe the signal's actual position.
- For a 3-lamp housing (red/amber/green in one case), box **only the single
  lit lamp's cell of the housing**, not the whole 3-lamp box — we need to
  know *which* lamp is lit, and one box covering all three lamps cannot
  encode that.

## 2. Signal classes — colour, ped/veh split, and edge cases

**REVISED 2026-09-27 after a reviewer found a real false-safe defect. Read
the "Why this changed" note below before anything else in this section --
the previous version of this rule was wrong, not just imprecise.**

The system's one hard rule (PLAN.md ground rules) is: **a false "safe to
cross" is the only unacceptable failure.** Every rule below is written to
protect that, even where it costs recall.

### Why this changed

This section used to say (verbatim): *"Pedestrian signal (walking-man icon)
vs vehicle signal: label with the same signal_red/signal_green classes — we
do not have separate classes for pedestrian vs vehicle signals."* That rule
was the root cause of a real false-safe path: src/fsm.py's SAFE_TO_CROSS
rule fired on ANY signal_green detection, and roughly four of the five
public signal datasets this project trains on are VEHICLE traffic lights
shot from inside a car (see scripts/download_signal_datasets.py's
docstring). A green light for CARS means traffic has right of way — it is
evidence of danger to a pedestrian, not permission to cross. Sharing one
class between the two meanings meant the model (and the FSM) could not tell
"you may walk" from "cars may go," and the concrete failure mode was: a
vehicle light turns green, cars approaching from the side or driving away
from the camera don't register as an approaching-TTC threat, and the system
announces "safe to cross now" exactly as traffic accelerates through the
junction. This is now fixed by splitting the taxonomy into four classes:
**ped_signal_red, ped_signal_green, veh_signal_red, veh_signal_green.**
Only ped_signal_green may ever produce a SAFE verdict; the other three are
all WAIT-only evidence in src/fsm.py, including veh_signal_red — a stopped
vehicle does not tell a pedestrian anything about turning traffic, a second
carriageway, or a stale observation, so it is deliberately NOT sufficient
for SAFE either.

- **Amber/yellow:** label as **ped_signal_red or veh_signal_red** (matching
  whether it's a pedestrian or vehicle signal head), never the green side of
  that pair, and never a third class. Amber means traffic is stopping but
  may still be moving — treating it as red is the conservative direction of
  the error. Do not invent a signal_amber class; PLAN.md fixes the class
  list at 12 and amber data is too sparse to support a 13th class properly.
- **Signal off / blank (no lamp lit):** **do not label it at all.** An unlit
  signal gives no evidence either way, and forcing it into red or green
  would teach the model a colour that was not actually shown. This does mean
  a real-world blank/broken signal will be invisible to the system — that is
  a known, accepted limitation, not a labelling bug.
- **Pedestrian signal (walking-man icon) vs vehicle signal:** label with the
  matching pair — **ped_signal_red/ped_signal_green** for a walking-man
  icon, **veh_signal_red/veh_signal_green** for a standard vehicle traffic
  light. These are now separate classes precisely because they mean
  different things to a pedestrian (see "Why this changed" above). If a
  frame has both a vehicle signal and a pedestrian signal showing
  **different** colours (this happens — pedestrian red can persist slightly
  after vehicle green, or vice versa), label **both boxes, each with its own
  true colour AND its own correct ped/veh class**. Do not average or guess a
  single answer for the frame; the model should see both signals as they
  actually are, and must be able to tell which one is which.
- **Ambiguous colour** (glare, low resolution, backlit sky, camera
  exposure has blown out the lamp to white): **do not label it as either
  colour.** Skip the box entirely rather than guess. A guessed green that is
  actually red is exactly the false-safe failure mode we are trying to
  avoid; an unlabelled signal just costs recall, which is the safe side to
  err on.
- **Ambiguous ped vs veh** (can't tell from the frame whether a signal head
  is a pedestrian walking-man signal or a vehicle light — e.g. too small,
  too far, unfamiliar signal design): **do not label it at all**, same
  reasoning as ambiguous colour. Guessing "pedestrian" on a vehicle light
  would recreate exactly the false-safe path this split exists to close.
- **When two lamps appear lit at once** (bad bulb, camera rolling shutter
  artifact): skip the box. This is not a real state a driver/pedestrian
  would treat as meaningful, and forcing a single label would be a guess.

## 3. Partially visible vehicles at the frame edge

**Proposal: label if at least 40% of the vehicle's expected extent is
visible in-frame**, judged by eye (front/back of a car, wheelbase of a
bike), not a formal calculation. Box only the visible portion — do not
extend the box off-canvas to where you imagine the rest of the vehicle is.

Below 40% visible (e.g. a sliver of a bumper at the frame edge), skip it.
Below that threshold the box is mostly a guess about the vehicle's true
size and aspect ratio, and inconsistent guesses between labellers are worse
than a missed detection.

## 4. Minimum pixel size

**Proposal: 10px minimum on the shorter box dimension for vehicles/person,
15px minimum on the shorter dimension for crosswalk and signal boxes.**

Signals and crosswalks get a higher floor because PLAN.md already flags
them as a known problem ("Signal too small... often 20-30px wide") — boxing
something at 6-8px teaches the model on near-noise and will not survive
resizing to training resolution (640 or 960) without collapsing to nothing.
If a signal or crosswalk is visible but smaller than this, skip it rather
than force a box.

## 5. Occlusion

**Proposal: label if at least 50% of the object would be visible if nothing
were in front of it**, again judged by eye using context (a car's roofline
and one wheel visible past a pole implies the rest of the car). Box the
visible extent only; do not draw through the occluder.

Below 50% occluded, skip it — same reasoning as the frame-edge rule: past
that point the box shape is mostly invented.

Exception: **do not apply this leniently to signals.** A half-occluded
signal head where the lit colour itself is not clearly readable must be
skipped regardless of the 50% rule (see section 2's "ambiguous colour"
rule) — occlusion of the housing is fine as long as the lamp colour is
unambiguous; occlusion of the *lamp* is not.

## 6. Crosswalk extent

**Proposal: box the painted zebra stripes only, not the full kerb-to-kerb
road width.**

Reasoning: the painted stripes have a hard, visually obvious edge that two
labellers will draw the same way. "Full road width between kerbs" requires
guessing where an unpainted kerb line actually is when it is not clearly
marked, which is exactly the inconsistency this document exists to prevent.
The system needs the crosswalk box to know *where a legal crossing point
is*, which is what the painted stripes mark — the unmarked road surface
next to it is not evidence of a designated crossing.

## 7. Groups / crowds of people

**Proposal: box each visible person individually up to a crowd of ~6-8
clearly separable people in a frame. Beyond that density, box only the
people relevant to the crossing decision** (i.e. anyone standing at or
walking through the crosswalk/kerb area the camera is watching), and skip
individually boxing a dense unrelated crowd in the background (e.g. a
market crowd on the far pavement, unrelated to the crossing).

Do not draw one big box around a cluster of people — that teaches the model
an incorrect object shape and breaks per-person counting, which the safety
logic may eventually want.

## 8. Countdown signals (signal_countdown, class 9) — PROPOSAL

**Status: DRAFT, more so than the rest of this document.** Unlike every
other class here, there is currently no dataset of any kind -- public or
our own -- with a single boxed signal_countdown example. This section is
therefore pure convention-setting for Phase 4 labelling, written before any
frame has actually been labelled against it. Expect it to need revision
once the team looks at real footage.

**What to box:** the numeric digit display itself (the LED/LCD panel
showing the countdown number), not the whole signal head and not the
pedestrian walking-man icon next to it if the two are physically separate
housings. This follows the same logic as section 1's signal-lamp rule: the
digit panel has a hard physical housing edge, while trying to box "the
countdown as a concept" (including the icon, the pole, the mounting bracket)
would reintroduce exactly the inter-labeller inconsistency section 1 exists
to avoid. If the digit panel and the walking-man icon share one integrated
housing (common on combined pedestrian signal units), box the digit panel
portion only, the same way a 3-lamp vehicle signal gets one box per lit
lamp cell, not one box for the whole housing.

**Unreadable digits:** if the numeral itself cannot be confidently read
(distance, glare, motion blur, a digit mid-transition between two numbers),
still box it as signal_countdown if you can tell with confidence that it
*is* a lit countdown display -- the class only asserts "there is a
countdown timer here", not "the timer reads N seconds". This differs from
the signal_red/signal_green ambiguous-colour rule in section 2, because
reporting the wrong colour is safety-relevant in the way this system uses
it (feeds a SAFE/WAITING decision) while reporting the wrong digit is not,
since (see below) this box is never used to read the actual number. If you
cannot even tell it is a countdown display at all (looks like unlit glass,
or too small to be sure it isn't noise), skip it, per section 4's minimum
size rule and section 9's default.

**CRITICAL — how a countdown interacts with the red/green rules, and with
the FSM:** a countdown is information, not permission. Concretely:

- **A countdown alone (no readable red/green signal in the same frame) is
  NOT sufficient evidence for anything.** Do not treat a lone countdown as
  implying "so the signal must be red" or "so the signal must be green" --
  label only what is actually visible, per section 2's core rule (do not
  guess a colour that was not shown). A countdown next to an ambiguous or
  unlit lamp gets no colour label, same as section 2's existing rule for an
  unlit or ambiguous lamp on its own.
- **A countdown running down next to a green signal must NOT be read as
  extending how long the system considers the road safe.** The intuitive
  but wrong idea is "green with 8 seconds left is safer to start crossing
  than green with 1 second left, so factor the countdown into the
  crossing-time budget." This project's src/fsm.py deliberately does not
  do that, and should not be extended to: fsm.py's _required_crossing_time()
  already asks "is there enough time to walk this crossing", derived from
  road width and walking speed (see fsm.py's own constants), not from how
  long a light will stay green. A pedestrian countdown at a real
  intersection describes the *vehicle* signal's timing in many
  installations, not a promise about how long it is safe to be mid-crossing
  once vehicles get a green in the cross direction -- treating it as a
  safety extension would be trusting a signal-timing inference this project
  has no way to verify, in the one system whose only hard rule is "a false
  SAFE is unacceptable" (PLAN.md's ground rule). A countdown reaching zero
  is, if anything, a reason for MORE caution (the phase is about to
  change), never less.
- **This document does not propose any new fsm.py rule that reads
  signal_countdown at all**, safe or otherwise. See src/fsm.py's own
  comments for where a *future*, carefully-scoped use might go (e.g.
  refusing to start a fresh SAFE verdict when a countdown is very close to
  zero, which is strictly more conservative and does not touch the
  affirmative SAFE rule at all) -- but with zero training data for this
  class, any such rule would be completely untestable today, so it stays a
  comment, not code.

## 9. When in doubt

**If you cannot decide confidently within a few seconds, skip the box and
move on.** A missing label costs the model some recall, which we can offset
with more data later. A wrong label (wrong class, guessed colour, invented
extent) actively teaches the model something false, and false signal labels
in particular feed directly into the one failure mode this whole project is
built to avoid (PLAN.md: "a false SAFE is the only unacceptable failure").
When in doubt, leave it out.

---

## Sign-off

Everyone who labels frames should read this document, raise disagreements,
and sign below once the team has agreed on a final version (edit the rules
above first, then sign against the agreed version — do not sign the draft
as-is if you want changes).

| Name | Agrees to this version | Date |
|------|------------------------|------|
|      |                        |      |
|      |                        |      |
|      |                        |      |
