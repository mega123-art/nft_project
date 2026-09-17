# Safe Road-Crossing Assistance System — Build Plan

College final-year project. Vision-based system that watches a road crossing
through a camera and gives audio guidance about when it is safe to cross.

## Ground rules for this project

- Runs on a **PC only**. No phone, no Raspberry Pi, no TFLite, no quantization.
- Python 3.10+, CPU inference is acceptable (8-12 fps is enough).
- Build and test **one phase at a time**. Do not scaffold the whole system
  up front. Each phase must run end to end before the next one starts.
- The default output of the system is always "wait". Only explicit positive
  evidence produces a "safe to cross" announcement. A false "safe" is the only
  unacceptable failure mode.
- Keep the decision logic as plain readable Python. No state-machine library.
  We have to explain every line in the viva.

## Target repo structure

```
road-crossing/
  PLAN.md
  requirements.txt
  data/
    raw_videos/          own phone footage (gitignored)
    frames/              extracted frames (gitignored)
    datasets/            downloaded + merged YOLO datasets (gitignored)
    data.yaml            unified class config
  models/
    best.pt              trained weights (gitignored, but keep one in releases)
  scripts/
    extract_frames.py
    remap_classes.py
    merge_datasets.py
    prelabel.py
    train.py
    evaluate.py
  src/
    detector.py          YOLO wrapper, returns typed detections
    safety.py            tracking history, approach detection, TTC
    fsm.py               crossing state machine
    audio.py             TTS + earcons, threaded, rate limited
    main.py              capture loop, wires everything together
  tests/
  logs/                  decision logs for evaluation (gitignored)
  report/                figures and metrics for the write-up
```

## Unified class list

Every dataset gets remapped to exactly these IDs. Nothing else.

| ID | Name | Source |
|----|------|--------|
| 0 | car | all datasets |
| 1 | bus | all datasets |
| 2 | truck | all datasets |
| 3 | motorcycle | all datasets (merge "bike", "scooter", "MotorBike") |
| 4 | autorickshaw | Indian datasets only (merge "Rikshaw", "Auto") |
| 5 | person | all datasets |
| 6 | crosswalk | Roboflow zebra crossing + our own footage |
| 7 | signal_red | Roboflow + our own footage |
| 8 | signal_green | Roboflow + our own footage |

Classes 6-8 will be weak after Phase 2 and are mainly fixed by our own
footage in Phase 4. That is expected. Do not try to fix it earlier.

---

## Phase 0 — Scaffold

**Goal:** empty repo that installs and runs.

1. `git init`, add a `.gitignore` covering `data/`, `models/*.pt`, `logs/`,
   `runs/`, `__pycache__/`, `*.mp4`.
2. `requirements.txt`:
   ```
   ultralytics
   opencv-python
   numpy
   pyttsx3
   pygame
   pyyaml
   ```
3. `src/main.py` that opens a webcam or a video file path from `argv`,
   shows the frame with `cv2.imshow`, exits on `q`. Nothing else.

**Done when:** `python src/main.py data/raw_videos/test.mp4` plays a video.

### Environment deviations (recorded 2026-09-17)

This machine ships only Python 3.14.4 (Ubuntu 26.04), so the dependency list
above needed two changes:

- `pygame` has no cp314 wheel. We use **`pygame-ce`** instead — drop-in,
  `import pygame` is unchanged.
- `pyttsx3` loads `libespeak-ng` via ctypes and espeak-ng is not installed.
  Phase 7 will need `sudo apt install espeak-ng`.

Also relevant later: no GPU on this box (training goes to Colab as planned),
and no `ffmpeg`, so Phase 4 frame extraction must use OpenCV.

---

## Phase 1 — Get the public datasets

**Goal:** one merged YOLO-format dataset on disk with our unified class IDs.

1. Download **Indian Roads Detection** from Roboflow Universe
   (`indian-road-dataset/indian-roads-detection`, ~2.6k images, 47 classes).
   Export as **YOLOv8**. This is the primary dataset — it has Autorickshaw,
   Zebra Crossing, Traffic Signal and Indian vehicle types all in one place.
2. Download the **Zebra Crossing** dataset
   (`project-5wrkt/zebra-crossing-inwkp`, ~300 images) as YOLOv8.
3. Write `scripts/remap_classes.py`:
   - Takes a source dataset dir and a JSON mapping `{old_id: new_id}`.
   - Rewrites every `.txt` label file with the new IDs.
   - **Drops** any annotation whose old ID is not in the mapping (we do not
     want Cattle, Lamp Post, Building, etc.).
   - Deletes images whose label file ends up empty.
   - Prints a per-class count before and after so we can sanity check.
4. Write `scripts/merge_datasets.py` that combines remapped datasets into
   `data/datasets/merged/{train,val}/{images,labels}` with a 85/15 split, and
   writes `data/data.yaml` with our 9 class names.

**Important:** the Roboflow class lists are messy and contain duplicates with
different capitalisation ("Bus stop" vs "Bus Stop", "Traffic POlice" vs
"Traffic Police"). Read the actual `data.yaml` from each download and build
the mapping from that file, not from assumptions.

**Done when:** `python scripts/merge_datasets.py` produces a valid dataset and
prints a class histogram. Verify by opening 5 random images with their boxes
drawn.

---

## Phase 2 — Baseline training

**Goal:** a model that detects Indian vehicles well and crossings/signals badly.

1. `scripts/train.py` wrapping Ultralytics:
   ```python
   model = YOLO("yolov8n.pt")
   model.train(data="data/data.yaml", epochs=60, imgsz=640, batch=16)
   ```
   Make epochs, imgsz and batch CLI args with these as defaults.
2. Train on Google Colab free T4 if no local GPU. Roughly 30-40 minutes.
3. Copy `best.pt` to `models/`.
4. `src/detector.py`: a `Detector` class that loads the weights and exposes
   `detect(frame) -> list[Detection]` where `Detection` is a dataclass with
   `cls_name, conf, x1, y1, x2, y2, track_id`. Use
   `model.track(frame, persist=True, conf=0.4)` so tracking comes free.
5. Wire the detector into `main.py` and draw boxes.

**Done when:** running on a road video draws correct boxes on cars, bikes and
rickshaws. Record the per-class mAP50 from the training run — this is the
"public data only" baseline number for the report.

---

## Phase 3 — Add DriveIndia or IDD for vehicle strength

**Goal:** better vehicle recall, especially at distance and in poor light.

1. Register for **DriveIndia** at the TiHAN IIT-Hyderabad portal
   (`tihan.iith.ac.in/TiAND.html`). It is already in YOLO format, 66,986
   images, 24 classes, including fog and rain. Alternatively **IDD-Detection**
   or **IDD-Lite** from `idd.insaan.iiit.ac.in` (needs format conversion).
2. Remap its vehicle and person classes into IDs 0-5. Ignore everything else.
3. Subsample — do not use all 67k images. Take ~8,000 stratified across the
   vehicle classes. Full DriveIndia will dominate the crossing/signal classes
   and make them worse.
4. Retrain. Compare per-class mAP50 against Phase 2 and record both.

**Note:** these are dashcam datasets shot from a car bonnet looking down the
road. Our camera looks *across* the road from the kerb. They help vehicle
detection only. They will not help crosswalk or signal detection, and may
slightly hurt it. If the crossing classes drop badly, reduce the subsample
size and retrain.

**Done when:** vehicle mAP50 improved over Phase 2 and the numbers for both
runs are saved in `report/`.

---

## Phase 4 — Our own footage

**Goal:** fix the crosswalk and signal classes for the pedestrian viewpoint.

This is the phase that actually makes the demo work. The public datasets have
almost no pedestrian-standing-at-kerb footage.

1. Record 20-30 minutes on a phone at chest height at signalized crossings.
   Cover: morning, evening, overcast, one night clip. Capture both red-with-
   traffic and green-with-clear-road cases. Negative examples matter.
2. `scripts/extract_frames.py` — one frame every 0.5 seconds, writes JPGs to
   `data/frames/`. ~3,600 frames from 30 minutes.
3. Hand-label ~150 frames in Roboflow with our 9 classes.
4. `scripts/prelabel.py` — run the Phase 3 model over the remaining frames
   with `save_txt=True, conf=0.3` to generate draft YOLO labels. Upload images
   + labels back to Roboflow and **correct** rather than draw. Roughly 5x
   faster.
5. Target 800-1200 corrected frames total.
6. Agree labelling conventions across the team **before** anyone starts:
   exact box edges on a signal head, whether partially visible vehicles at the
   frame edge count, minimum size for a distant crossing. Write these into
   `data/LABELLING.md`. Inconsistency between teammates is the main cause of
   poor accuracy in projects like this.
7. Merge into the training set and retrain. Weight our own data by duplicating
   it 2-3x so it is not drowned out.

**Done when:** crosswalk and signal_red/signal_green mAP50 are usable (> 0.6)
on a held-out set of our own footage that was never trained on.

---

## Phase 5 — Motion and time-to-contact

**Goal:** know which vehicles are approaching and how soon they arrive.

`src/safety.py`:

- Keep a `deque(maxlen=5)` of bounding-box widths per `track_id`.
- Smooth the widths before differentiating, or noise destroys the estimate.
- A single camera gives no depth. Estimate time-to-contact from box growth:
  for a fixed real-world width, image width is proportional to 1/distance, so
  `TTC ≈ w / (dw/dt)`.
- Shrinking box means the vehicle is receding — ignore it entirely.
- Also compute lateral drift of the box centre, to tell a car crossing our
  path from one parked at the kerb.
- Expose `min_ttc()` across all currently approaching vehicles, and
  `None` when nothing is approaching.
- Drop tracks not seen for 15 frames.

**Done when:** an overlay on the video shows a TTC number above each vehicle
and it visibly decreases as a car approaches. Eyeball it against a few clips
where you can count seconds manually.

---

## Phase 6 — Decision state machine

**Goal:** turn detections into one of a small set of verdicts.

`src/fsm.py`. States:
`SEARCHING -> AT_CROSSING -> WAITING -> SAFE_TO_CROSS -> CROSSING -> DONE`

Rules, in order:

```
no crosswalk detected           -> SEARCHING,  "looking for a crossing"
signal_red present              -> WAITING,    "wait, signal is red"
min_ttc is not None and < 5.0   -> WAITING,    "wait, vehicle approaching"
signal_green and road is clear  -> SAFE,       "safe to cross now"
otherwise                       -> WAITING,    "unclear, please wait"
```

Plus:

- **Hysteresis.** Require the same verdict for 8 consecutive frames (~0.3 s)
  before the state actually changes. Store a small vote buffer.
- **Crossing time check.** Estimated crossing time = road width / 1.0 m/s
  walking speed, plus a 3 second margin. Only declare SAFE if the clear gap
  exceeds that. Road width can be a config constant for now.
- **Fail safe.** Any exception, any low-confidence frame, any camera-tilt
  rejection -> WAITING. Never SAFE by default.
- If the crosswalk box is detected, also report its horizontal offset from
  frame centre so we can say "crossing is to your right".

**Done when:** the state and reason are drawn on the video and behave sanely
across all test clips. Test the red-light clip specifically: it must never
enter SAFE.

---

## Phase 7 — Audio

`src/audio.py`:

- `pyttsx3` for speech, running on a **separate thread** or the video will
  stutter. Use a queue with `maxsize=1` and drop stale messages.
- `pygame.mixer` for earcons: a rising two-tone for safe, a repeated low pulse
  for danger. Earcons are much faster than speech and matter more for urgency.
- Rate limit: do not repeat the same announcement within 2 seconds, and do not
  announce anything more than once per 2 seconds overall.
- Speak only on state *change*, not every frame.

**Done when:** the system talks through a full clip without stuttering the
video and without spamming.

---

## Phase 8 — Evaluation and report material

`scripts/evaluate.py` and SQLite logging in `main.py`.

Log every frame's decision with timestamp, state, confidence, min_ttc and the
detected classes to `logs/decisions.db`.

Metrics to produce for the report:

- **False-safe rate** — fraction of frames where the system said SAFE but
  ground truth says unsafe. Report this first and most prominently. Target 0.
- Per-class mAP50 at each of Phase 2, 3, 4 (shows the contribution of our own
  data — this is the strongest graph in the report).
- End-to-end latency from frame capture to audio start. Target < 300 ms.
- TTC error against manually annotated ground truth on ~20 vehicle tracks.
- Frame rate on the test machine.

Also generate: confusion matrix, PR curves (Ultralytics produces these in
`runs/`), and 6-8 annotated screenshots showing each state.

## Things that commonly go wrong

- **Class imbalance.** DriveIndia will swamp the crossing classes. Subsample.
- **Signal too small.** A pedestrian signal is often 20-30 px wide. If
  detection is poor, raise `imgsz` to 960 for training and inference before
  reaching for a two-stage crop-and-classify approach.
- **Label leakage between splits.** Frames extracted from the same video are
  near-identical. Split our own data **by video file**, not by frame, or the
  validation score will be meaningless.
- **Speech blocking the loop.** Always threaded.
- **`persist=True` forgotten** in `model.track()` — track IDs reset every
  frame and TTC becomes garbage.

## Framing for the report

Existing Indian driving datasets (IDD, DriveIndia) are vehicle-mounted and
capture a driver's viewpoint. This project contributes a small
pedestrian-viewpoint dataset for crossing assistance and quantifies the
accuracy gain it provides over public data alone.

State the limitation explicitly: this is an assistive aid, not a replacement
for a white cane or the user's own judgement. Monocular TTC is an estimate,
not a measurement.
