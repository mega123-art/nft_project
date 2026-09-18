# Phase 8 ground-truth format

`scripts/evaluate.py` needs two things that do not exist anywhere else in
this repo: a human saying which frames were actually safe/unsafe to cross,
and a human's own stopwatch-or-eyeball estimate of a vehicle's real
time-to-contact. Both are manual annotation. This file is the schema for
both, written down once so annotation stays consistent and evaluate.py has
something fixed to parse.

Neither file exists yet for the current sample clips (`data/raw_videos/`
is elevated CCTV of non-Indian roads with sparse detections -- domain
mismatch, not something worth hand-labelling). evaluate.py checks for these
files and reports the affected metric as **NOT COMPUTED** when they are
missing, rather than inventing a number. See report/evaluation.md's own
output for exactly which metrics that affects right now.

## 1. Frame-range safe/unsafe labels

**Path:** `data/ground_truth/frame_labels.csv`

One row per contiguous range of frames a human has judged, for a given
video. Ranges are inclusive of both endpoints, and referenced by the same
`video_source` string `main.py`/`decision_log.py` stamp onto each logged
row (i.e. whatever path or `"camera 0"` was passed on the command line --
match it exactly, including any relative path prefix).

```csv
video_source,start_frame,end_frame,label
car-detection.mp4,1,120,unsafe
car-detection.mp4,121,400,safe
```

- `label` is exactly `safe` or `unsafe` (lowercase). There is no "unclear"
  label on purpose -- if a human annotator genuinely cannot tell, that
  frame range should simply be left out of the file. evaluate.py only ever
  scores frames that appear in this file; anything absent is treated as
  "no ground truth for this frame," never as an implicit label.
- Ranges must not overlap for the same video. evaluate.py does not attempt
  to resolve a conflict; overlapping ranges are a labelling bug to fix by
  hand.
- `unsafe` means: a human would not have crossed the road at that moment
  (a vehicle is close/approaching, the signal is red, etc). `safe` means
  the human judges it genuinely fine to cross.

**How this is used:** the false-safe rate is computed only over frames
that fall inside some row of this file. For each such frame, evaluate.py
looks up what `decisions.db` recorded as the *committed* `state` for that
`(video_source, frame_index)` and counts it as a false-safe if the label is
`unsafe` and the state is `SAFE_TO_CROSS`.

`scripts/annotate_ground_truth.py` (see below) is a keypress-driven helper
that writes this file directly; hand-editing the CSV works too.

## 2. TTC observations

**Path:** `data/ground_truth/ttc_observations.csv`

One row per manually-timed observation of a real vehicle's true
time-to-contact, matched to the frame at which the system logged its own
`min_ttc` estimate for that scene.

```csv
video_source,frame_index,true_ttc_s
car-detection.mp4,88,4.2
car-detection.mp4,140,2.1
```

- `frame_index` must match a `frame_index` actually present in
  `decisions.db` for that `video_source` -- evaluate.py does not
  interpolate between frames, on the theory that a human's manual timing
  is already an estimate and inventing an interpolated match would compound
  two estimates into a number no more trustworthy than either alone.
- `true_ttc_s` is the human's estimate of the real, physical
  time-to-collision at that instant (e.g. counted against a stopwatch and a
  known camera-to-crossing distance, or read off a dashcam clip with known
  road markings). How exactly to produce this number is a per-clip judgement
  call; write down the method used alongside the file (e.g. in a commit
  message or a short comment column) so the number can be sanity-checked
  later.
- PLAN.md asks for ~20 annotated vehicle tracks. This file is one row per
  *observation* (a single instant), not one row per track, since
  `decisions.db` only logs a scene-wide `min_ttc`, not a per-track history.
  Annotate the moment(s) that vehicle is the closest/only approaching
  vehicle in frame, so `min_ttc` at that frame is unambiguously about that
  track. Aim for ~20 observations spread across distinct vehicles/clips.

**How this is used:** evaluate.py joins this file against `decisions.db`
on `(video_source, frame_index)`, computes `predicted_ttc - true_ttc_s` per
row (using the frame's logged `min_ttc`, which is `NULL`/`None` if nothing
was assessed as approaching at that instant -- such rows are reported
separately as "system reported no TTC" rather than silently dropped or
counted as zero error), and reports the signed and absolute error
distribution.

## Producing `frame_labels.csv` in practice

`scripts/annotate_ground_truth.py <video> --video-source <name>` steps
through a clip frame by frame with OpenCV and lets you mark unsafe ranges
with keypresses, writing `data/ground_truth/frame_labels.csv` directly.
See that script's own module docstring for the exact keys. It is a
convenience, not a requirement -- hand-writing the CSV per the format above
works identically as far as evaluate.py is concerned.
