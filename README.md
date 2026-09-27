# Safe Road-Crossing Assistance System

A camera watches a road crossing and says out loud when it is safe to cross.
Built for people who cannot easily judge traffic themselves — visually
impaired and elderly pedestrians.

Runs on an ordinary PC. CPU inference. No phone, no Raspberry Pi.

---

## The one rule everything else follows

> **The default answer is always "wait". Only explicit positive evidence
> produces "safe to cross".**

A wrong "wait" is annoying. A wrong "safe" can kill someone. These two
errors are not equally bad, so the system is deliberately **asymmetric**
everywhere: evidence of danger is accepted cheaply, evidence of safety must
be strong and sustained.

If you read nothing else here, read that. Every threshold below is a
consequence of it, and "why is this number asymmetric?" is the question this
design exists to answer.

---

## How it works, end to end

```
camera / video file
        │
        ▼
  ┌───────────┐   YOLOv8, 13 classes, 960px
  │ detector  │   → boxes + confidence
  └─────┬─────┘
        │
        ▼
  ┌───────────┐   ByteTrack: same car keeps the same ID
  │  tracker  │   across frames
  └─────┬─────┘
        │
        ▼
  ┌───────────┐   how fast is each box GROWING?
  │    TTC    │   → seconds until it reaches the camera
  └─────┬─────┘
        │
        ▼
  ┌───────────┐   7 rules, in order, plain Python
  │  decision │   → SEARCHING / WAITING / SAFE_TO_CROSS
  └─────┬─────┘
        │
        ▼
  ┌───────────┐   slow to permit, instant to withdraw
  │ hysteresis│
  └─────┬─────┘
        │
        ├──────────────► speech + earcons
        └──────────────► SQLite log (one row per frame)
```

---

## The 13 classes

| | class | used for |
|---|---|---|
| 0 | `car` | hazard, TTC |
| 1 | `bus` | hazard, TTC |
| 2 | `truck` | hazard, TTC |
| 3 | `motorcycle` | hazard, TTC |
| 4 | `autorickshaw` | hazard, TTC |
| 5 | `person` | context |
| 6 | `crosswalk` | **required** before any decision is made |
| 7 | `ped_signal_red` | "do not walk" → WAIT |
| 8 | `ped_signal_green` | **the only class that can permit crossing** |
| 9 | `veh_signal_red` | neutral (see below) |
| 10 | `veh_signal_green` | **danger** → WAIT |
| 11 | `signal_countdown` | not used by the decision logic |
| 12 | `signal_unknown` | never used; exists only to protect training |

### Why pedestrian and vehicle signals are separate classes

This is the most important design decision in the project, and it came from
a bug.

Originally there were just `signal_red` and `signal_green`, shared between
pedestrian signals and vehicle traffic lights. A reviewer asked a simple
question: *doesn't a green signal then mean "safe to cross"?*

It did — and that was wrong. **A green light for cars means traffic has
right of way.** For a pedestrian it is the opposite of permission. Since
four of the five signal datasets are vehicle traffic lights, `signal_green`
mostly meant "cars are moving", and the system turned that into
"safe to cross now".

The failure needed no detector error at all:

1. Vehicle light turns green
2. Cars ahead drive **away** from the camera → tracked as not approaching
3. Road therefore reads as clear
4. Green signal + clear road → **"safe to cross"**
5. Said to a blind pedestrian as traffic accelerates through the junction

A road that has just turned green for cars is often momentarily empty, which
is exactly when this misfires. The classes were split so the decision logic
can tell *who a signal is talking to*.

### Why `signal_unknown` exists

Some datasets label signals without saying which lamp is lit. We cannot give
those a colour — guessing is how false-safes happen.

But **deleting a label does not delete the object from the image.** YOLO
learns unlabelled pixels as background, so dropping 1,490 signal boxes would
have taught the model that signal-shaped things *are* background — damaging
the classes we most need. `signal_unknown` absorbs them. The decision logic
never reads it.

---

## The decision rules

Checked **in order**. The first one that matches wins. Plain `if` statements
in [`src/fsm.py`](src/fsm.py) — no state-machine library, because every
branch has to be explainable out loud.

| # | condition | result |
|---|---|---|
| 1 | no `crosswalk` visible | `SEARCHING` |
| 2 | `ped_signal_red` (or an ambiguous legacy red) | `WAITING` |
| 2b | `veh_signal_green` | `WAITING` |
| 3 | a vehicle is turning across the crossing | `WAITING` |
| 4 | some vehicle's TTC < 5.0s | `WAITING` |
| 5 | `ped_signal_green` ≥ 0.6 **and** road clear | **`SAFE_TO_CROSS`** |
| 5b | no pedestrian signal exists **and** road clear for 24 cycles | **`SAFE_TO_CROSS`** |
| 6 | anything else | `WAITING` |

Only rules 5 and 5b can ever permit crossing. Every rule above them is
checked first, so none of them can be bypassed.

### "Road clear" means all of these

- **No unassessable vehicle in frame.** A vehicle the tracker cannot yet
  judge is not evidence of safety *or* danger — it blocks a SAFE verdict
  rather than being assumed harmless.
- **Either no vehicle is approaching at all**, or the nearest one is further
  away in time than the crossing actually takes to walk.
- Crossing time is `road_width / walking_speed + margin` = **13s** by
  default, not the 5s alarm threshold. 5s answers "is something about to
  hit me"; it says nothing about whether there is time to walk across.

### Why `veh_signal_red` is neutral

When a pedestrian signal turns green, the vehicle signal is red — that *is*
the pedestrian phase. An earlier version treated `veh_signal_red` as a
reason to wait, which meant `ped_signal_green + veh_signal_red` resolved to
WAIT, making SAFE unreachable at any real intersection no matter how good
the detector got.

So `veh_signal_red` neither permits nor blocks. It cannot permit, because
stopped cars are not permission to walk — the system **cannot tell which
approach a signal head governs**, and a red facing the cross-direction means
the traffic that would hit you has green.

### Rule 5b: crossings with no pedestrian signal

Most Indian crossings have no walking-man signal at all — just painted
stripes and vehicle lights. Rule 5 alone would hunt forever for a signal
that does not physically exist, and sit in WAITING. Safe, but useless.

Rule 5b permits crossing on **direct observation** instead: no vehicle
approaching, nothing unassessable, enough time to walk — sustained for **24
consecutive decision cycles (~3 seconds)**, three times the normal
hysteresis.

The long window is the honest part. The camera sees a finite field of view,
so *"no vehicle visible"* is a weaker claim than *"no vehicle coming"* — a
car can be just outside frame or hidden behind a bus. Three seconds of
continuously clear road means a vehicle entering from outside the frame has
had time to appear and be tracked before any verdict is given.

**Rule 5b stands down if any `ped_signal_*` box is present at any
confidence.** Without that guard, a pedestrian green seen at 0.45 would fail
rule 5's 0.6 gate and then quietly succeed through 5b — laundering weak
evidence into a SAFE verdict. A crossing that *has* a signal must be decided
by that signal.

---

## Time-to-contact, without a calibrated camera

A box grows as a vehicle approaches. Box width `w` relates to distance `D`
as `w = k/D` — a **hyperbola**, so fitting `w` against time in a straight
line is biased, and biased *optimistic* (it thinks you have more time than
you do). That is the forbidden direction.

Fitting `u = 1/w` instead is **exactly linear**, and extrapolating to
`u = 0` gives the moment of contact. This cut the systematic error from
~0.3s optimistic to under 0.005s.

**Noise floor.** Boxes jitter by a pixel or two even on a parked car, and
that jitter scales with box size — so one fixed threshold cannot work for a
small distant box and a large near one. Each track gets its own horizon
`TTC_horizon = w·T/δ`, beyond which apparent motion is indistinguishable
from jitter.

**Three states per vehicle:** `APPROACHING`, `NOT_APPROACHING`, `UNKNOWN`.
`UNKNOWN` matters — an early version where any unassessed vehicle blocked
SAFE meant a single parked car blocked crossing forever. A vehicle moving
laterally across the crossing is reported `UNKNOWN` rather than
`NOT_APPROACHING`, because a turning vehicle is a real hazard that sideways
motion alone would dismiss.

---

## The asymmetries, collected

This is the table to look at if you want the design in one place.

| | evidence of **danger** | evidence of **safety** |
|---|---|---|
| confidence needed | any | ≥ 0.6 |
| frames needed | 1 — instant | 8 (or 24 unsignalled) |
| unknown vehicle | blocks SAFE | never assumed harmless |
| ambiguous signal | treated as danger | never treated as permission |
| amber light | counts as red | — |

Leaving SAFE is immediate on the very next frame. Entering it is slow.

---

## Running it

```bash
# watch it work on a video
.venv/bin/python src/main.py data/raw_videos/clip.mp4 --imgsz 960 --detect-every 2

# measure speed, no window
.venv/bin/python src/main.py clip.mp4 --headless --max-frames 300

# webcam
.venv/bin/python src/main.py
```

`q` quits.

| flag | meaning |
|---|---|
| `--imgsz` | inference resolution — **must match training (960)** |
| `--detect-every N` | run the detector every Nth frame (default 2) |
| `--weights` | model file, default `models/best.pt` |
| `--mute` | no speech |
| `--headless` | no window; prints fps |
| `--save-frames DIR` | dump annotated frames |
| `--log-db` / `--no-log` | per-frame SQLite decision log |

**`--imgsz` matters more than it looks.** A model trained at 960 and run at
1280 once detected *zero* crosswalks — and since crosswalk is rule 1, the
system sat in `SEARCHING` forever. Train and inference resolution must
match.

**`--detect-every` skips detection, not safety.** The tracker and the
decision rules do **not** run on skipped frames. Re-feeding an identical box
with a later timestamp would inject fake zero-growth into the TTC fit and
make vehicles look slower than they are — optimistic, the forbidden
direction. Skipping raised throughput from 9.4 to 16.5 fps.

---

## Layout

```
src/
  main.py         capture loop, frame skipping, wiring
  detector.py     YOLOv8 + ByteTrack wrapper
  safety.py       tracking, TTC, approach assessment
  fsm.py          the 7 rules + hysteresis
  audio.py        threaded speech + earcons
  decision_log.py per-frame SQLite logging
scripts/          dataset download, remap, merge, pseudo-label, train, evaluate
tests/            117 tests
data/LABELLING.md labelling conventions (read before annotating anything)
PLAN.md           the original build plan
```

Audio runs on its own thread so speech can never stall the vision loop, and
transitions **into** WAITING bypass the rate limiter — a warning always
plays.

---

## Honest limitations

Written plainly, because a safety system that hides its limits is worse than
one that states them.

- **The false-safe rate has not been measured.** It needs ground-truth
  labelled footage that reaches a SAFE verdict. This is the headline number
  and it does not exist yet.
- **Crosswalk detection does not generalise as well as its score suggests.**
  0.932 mAP50 on the validation split, but close to zero detections on real
  street footage from another region. A validation split can hide a domain
  gap completely — and since crosswalk is rule 1, losing it stops
  everything.
- **Signals are small.** On real night footage, raising inference resolution
  moved `signal_green` from 0.14 → 0.26 → 0.51 confidence (960 → 1280 →
  1920) — still under the 0.6 gate, at triple the compute. Distance and
  darkness are real limits.
- **Crosswalk markings are invisible at night** under traffic. Both effects
  above were measured on real footage, not assumed.
- **The camera sees one field of view.** "No vehicle visible" is not "no
  vehicle coming" — the reason rule 5b needs a long sustained window.
- **`road_width` is a 10m placeholder**, not measured per crossing.
- **An unlit or broken signal is invisible** to the system by design. We do
  not guess a colour that was not shown.
- **`ped_signal_green` is trained on ~769 instances** against
  `veh_signal_green`'s 5,453. The imbalance falls the safe way — confusing
  pedestrian green *as* vehicle green causes a false WAIT — but recall on
  the class that permits crossing is the thing to watch.
