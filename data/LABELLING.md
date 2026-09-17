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

## 2. signal_red vs signal_green — colour and edge cases

The system's one hard rule (PLAN.md ground rules) is: **a false "safe to
cross" is the only unacceptable failure.** Every rule below is written to
protect that, even where it costs recall.

- **Amber/yellow:** label as **signal_red**, never signal_green, and never a
  third class. Amber means traffic is stopping but may still be moving —
  treating it as red is the conservative direction of the error. Do not
  invent a signal_amber class; PLAN.md fixes the class list at 9 and amber
  data is too sparse to support a 10th class properly.
- **Signal off / blank (no lamp lit):** **do not label it at all.** An unlit
  signal gives no evidence either way, and forcing it into red or green
  would teach the model a colour that was not actually shown. This does mean
  a real-world blank/broken signal will be invisible to the system — that is
  a known, accepted limitation, not a labelling bug.
- **Pedestrian signal (walking-man icon) vs vehicle signal:** label with the
  same signal_red/signal_green classes — we do not have separate classes for
  pedestrian vs vehicle signals. If a frame has both a vehicle signal and a
  pedestrian signal showing **different** colours (this happens — pedestrian
  red can persist slightly after vehicle green, or vice versa), label
  **both boxes with their own true colour**. Do not average or guess a
  single answer for the frame; the model should see both signals as they
  actually are.
- **Ambiguous colour** (glare, low resolution, backlit sky, camera
  exposure has blown out the lamp to white): **do not label it as either
  colour.** Skip the box entirely rather than guess. A guessed green that is
  actually red is exactly the false-safe failure mode we are trying to
  avoid; an unlabelled signal just costs recall, which is the safe side to
  err on.
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

## 8. When in doubt

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
