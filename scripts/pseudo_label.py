"""
Fix for a training-data defect: some source datasets (e.g. zebra_crossing,
which annotates ONLY the crosswalk) contain pedestrians, cars and other
traffic that carry no label at all. YOLO treats every unlabelled region as
background, so those images actively teach the model that a person standing
at a kerb is background. This script runs a COCO-pretrained YOLO model over
a dataset and APPENDS pseudo-labels for the object classes that dataset
never annotated, leaving every existing (human-drawn) box untouched.

Deliberately NOT hardcoded to zebra_crossing: it walks every split dir under
the given dataset directory that has images/ and labels/ subfolders, so it
works on any dataset with the same layout.

Only COCO classes we actually care about are kept (see COCO_TO_UNIFIED
below); everything else COCO detects (train, boat, dog, ...) is discarded.
Per the project's decision, COCO's "bicycle" is folded into our
"motorcycle" id, same as the Roboflow "Cycle" class was in remap_classes.py.

Safety, matching remap_classes.py's convention: a real (non-dry-run) pass
writes a marker file (.pseudo_labeled) and refuses to run again on the same
dataset directory, because a second pass would append every pseudo box a
second time (on top of the first pass's own pseudo boxes, which are
indistinguishable from human boxes once written -- there would be nothing
left to de-dupe against). --dry-run never writes the marker and can be run
as many times as you like.

De-duplication: a candidate pseudo box is skipped if its IoU with an
EXISTING box of the same class exceeds --iou-dedup (default 0.5). This
only matters for datasets that already have partial annotations of a class
this script also produces; for zebra_crossing (crosswalk-only) it will
never fire, since no existing box is ever class 0/1/2/3/5 -- but the check
is here so this script is safe to use on datasets that aren't purely
single-class.

--dry-run changes nothing and additionally reports what --conf 0.25 and
0.5 would have added, next to whatever --conf was actually passed, so the
threshold's sensitivity is visible before committing to a value. This
extra reporting reuses ONE inference pass per image at the lowest of the
three thresholds, then filters by confidence in Python for the higher
ones -- it is a fast approximation (NMS itself is not exactly recomputed
per threshold) good enough for a sensitivity check, not a substitute for
actually rerunning at a different --conf if you change the default.
"""

import argparse
import glob
import os

from ultralytics import YOLO

MARKER_NAME = ".pseudo_labeled"

# our unified 9-class list, in ID order, used only for printing readable
# per-class counts.
UNIFIED_NAMES = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "autorickshaw",
    "person",
    "crosswalk",
    "signal_red",
    "signal_green",
]

# COCO class name -> unified class id. Anything not listed here (train,
# boat, dog, traffic light, ...) is ignored. There is no autorickshaw in
# COCO -- these are UK/European street scenes, so that is expected, not a
# bug. "bicycle" folds into "motorcycle" per the team's existing Cycle
# convention.
COCO_TO_UNIFIED = {
    "person": 5,
    "car": 0,
    "bus": 1,
    "truck": 2,
    "motorcycle": 3,
    "bicycle": 3,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_split_dirs(dataset_dir):
    """Any immediate subdirectory that has both images/ and labels/ under
    it (train, valid, val, test, ...) -- not assuming any particular split
    naming, so this works on any YOLO-layout dataset."""
    dirs = []
    for entry in sorted(os.listdir(dataset_dir)):
        split_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(os.path.join(split_path, "images")) and os.path.isdir(
            os.path.join(split_path, "labels")
        ):
            dirs.append(split_path)
    return dirs


def find_pairs(split_dir):
    """(image_path, label_path) for every image in this split. label_path
    may not exist yet -- that's fine, it gets created when we append."""
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    pairs = []
    for image_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if os.path.splitext(image_path)[1] not in IMAGE_EXTENSIONS:
            continue
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        pairs.append((image_path, label_path))
    return pairs


def parse_label_line(line):
    """Return (cls_id, (x1, y1, x2, y2)) normalised box, from either a
    plain YOLO bbox line (cls cx cy w h) or a YOLO-seg polygon line (cls
    followed by an even number of x,y coords) -- some source datasets
    (zebra_crossing) mix both. Polygons are reduced to their bounding box
    for IoU comparison only; the original line is never rewritten."""
    parts = line.split()
    cls_id = int(parts[0])
    coords = [float(v) for v in parts[1:]]
    if len(coords) == 4:
        cx, cy, w, h = coords
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    else:
        xs = coords[0::2]
        ys = coords[1::2]
        box = (min(xs), min(ys), max(xs), max(ys))
    return cls_id, box


def read_existing_boxes(label_path):
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                boxes.append(parse_label_line(line))
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_classes(pairs):
    counts = {}
    for _, label_path in pairs:
        for cls_id, _ in read_existing_boxes(label_path):
            counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def print_counts(title, counts):
    print(f"\n{title}")
    if not counts:
        print("  (no annotations found)")
        return
    for cls_id in sorted(counts):
        name = UNIFIED_NAMES[cls_id] if 0 <= cls_id < len(UNIFIED_NAMES) else "?"
        print(f"  id {cls_id:>3}  {name:<14}  {counts[cls_id]}")


def detect_raw(model, image_path, min_conf):
    """Run the COCO model once at min_conf, return every detection that
    maps to a unified class as (unified_cls_id, normalised_box, conf)."""
    results = model.predict(image_path, conf=min_conf, verbose=False)[0]
    boxes = results.boxes
    detections = []
    if boxes is None:
        return detections
    img_h, img_w = results.orig_shape
    for box in boxes:
        coco_id = int(box.cls[0])
        coco_name = model.names[coco_id]
        if coco_name not in COCO_TO_UNIFIED:
            continue
        unified_id = COCO_TO_UNIFIED[coco_name]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        norm_box = (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)
        detections.append((unified_id, norm_box, conf))
    return detections


def filter_and_dedup(raw_detections, existing_boxes, conf_threshold, iou_dedup):
    """Given raw detections (already at some low conf) and an image's
    existing boxes, apply a confidence threshold and skip anything whose
    IoU with an existing box of the SAME class exceeds iou_dedup. Returns
    (kept, n_skipped_iou)."""
    kept = []
    n_skipped = 0
    for cls_id, box, conf in raw_detections:
        if conf < conf_threshold:
            continue
        duplicate = False
        for ex_cls_id, ex_box in existing_boxes:
            if ex_cls_id == cls_id and iou(box, ex_box) > iou_dedup:
                duplicate = True
                break
        if duplicate:
            n_skipped += 1
        else:
            kept.append((cls_id, box))
    return kept, n_skipped


def box_to_yolo_line(cls_id, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Append COCO-pretrained pseudo-labels for classes a dataset never annotated"
    )
    parser.add_argument(
        "dataset_dir",
        help="dataset directory in YOLO layout (images/ and labels/ under each split dir)",
    )
    parser.add_argument("--weights", default="yolov8m.pt", help="COCO-pretrained weights (default yolov8m.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold used for the real run (default 0.4)")
    parser.add_argument(
        "--report-thresholds",
        nargs="*",
        type=float,
        default=[0.25, 0.5],
        help="extra conf thresholds to report on during --dry-run, for a sensitivity check (default 0.25 0.5)",
    )
    parser.add_argument(
        "--iou-dedup",
        type=float,
        default=0.5,
        help="skip a pseudo box whose IoU with an existing same-class box exceeds this (default 0.5)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be added without changing anything")
    args = parser.parse_args()

    marker_path = os.path.join(args.dataset_dir, MARKER_NAME)
    if not args.dry_run and os.path.isfile(marker_path):
        print(f"error: {marker_path} exists, meaning this dataset was already pseudo-labeled.")
        print("running it again would append every pseudo box a second time, with no way to")
        print("tell old pseudo boxes from human boxes to de-dupe against.")
        print("re-extract/re-download the dataset first if you need to redo this.")
        raise SystemExit(1)

    split_dirs = find_split_dirs(args.dataset_dir)
    if not split_dirs:
        print(f"error: no split dirs with images/ and labels/ found under {args.dataset_dir}")
        raise SystemExit(1)

    all_pairs = []
    for split_dir in split_dirs:
        pairs = find_pairs(split_dir)
        print(f"{split_dir}: {len(pairs)} images")
        all_pairs.extend(pairs)

    before_counts = count_classes(all_pairs)
    print_counts("BEFORE (existing annotations):", before_counts)

    print(f"\nloading {args.weights} ...")
    model = YOLO(args.weights)

    coco_wanted = sorted(set(COCO_TO_UNIFIED))
    print(f"keeping only COCO classes: {coco_wanted} (everything else COCO detects is discarded)")

    if args.dry_run:
        thresholds = sorted(set([args.conf] + list(args.report_thresholds)))
        lowest = thresholds[0]
        print(f"\nrunning {args.weights} at conf={lowest} (lowest requested threshold) over {len(all_pairs)} images...")

        per_image_raw = []
        for image_path, label_path in all_pairs:
            existing = read_existing_boxes(label_path)
            raw = detect_raw(model, image_path, lowest)
            per_image_raw.append((existing, raw))

        for t in thresholds:
            added_counts = {}
            skipped_total = 0
            images_gained_person = 0
            for existing, raw in per_image_raw:
                kept, n_skipped = filter_and_dedup(raw, existing, t, args.iou_dedup)
                skipped_total += n_skipped
                had_person_before = any(c == 5 for c, _ in existing)
                gains_person = any(c == 5 for c, _ in kept)
                if gains_person and not had_person_before:
                    images_gained_person += 1
                for cls_id, _ in kept:
                    added_counts[cls_id] = added_counts.get(cls_id, 0) + 1

            marker = "  <-- --conf default" if t == args.conf else ""
            print(f"\n=== dry-run at conf={t}{marker} ===")
            print_counts(f"would ADD (new pseudo-labels at conf={t}):", added_counts)
            print(f"  skipped as IoU-duplicate of an existing same-class box: {skipped_total}")
            print(f"  images that would gain a first person box: {images_gained_person} / {len(all_pairs)}")

        print("\n--dry-run: no files were modified.")
        return

    # real run: single pass at args.conf
    print(f"\nrunning {args.weights} at conf={args.conf} over {len(all_pairs)} images...")
    added_counts = {}
    skipped_total = 0
    images_gained_person = 0
    for image_path, label_path in all_pairs:
        existing = read_existing_boxes(label_path)
        raw = detect_raw(model, image_path, args.conf)
        kept, n_skipped = filter_and_dedup(raw, existing, args.conf, args.iou_dedup)
        skipped_total += n_skipped

        had_person_before = any(c == 5 for c, _ in existing)
        gains_person = any(c == 5 for c, _ in kept)
        if gains_person and not had_person_before:
            images_gained_person += 1

        if kept:
            new_lines = [box_to_yolo_line(cls_id, box) for cls_id, box in kept]
            with open(label_path, "a") as f:
                f.write("\n".join(new_lines) + "\n")
            for cls_id, _ in kept:
                added_counts[cls_id] = added_counts.get(cls_id, 0) + 1

    print_counts("ADDED (pseudo-labels appended):", added_counts)
    print(f"\nskipped as IoU-duplicate of an existing same-class box: {skipped_total}")
    print(f"images that gained a first person box: {images_gained_person} / {len(all_pairs)}")

    after_counts = count_classes(all_pairs)
    print_counts("AFTER (existing + pseudo annotations):", after_counts)

    with open(marker_path, "w") as f:
        f.write(f"pseudo_labeled with {args.weights} at conf={args.conf}, do not run pseudo_label.py on this directory again\n")
    print(f"\nwrote marker {marker_path}")
    print("done.")


if __name__ == "__main__":
    main()
