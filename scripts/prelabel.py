"""
Phase 4 step 4: run the trained model over extracted frames to produce draft
YOLO labels for Roboflow, so the team corrects boxes instead of drawing every
one from scratch.

Writes one .txt label file per image (YOLO format: "cls_id cx cy w h",
normalised 0-1), plus a data.yaml snippet in the output directory so Roboflow
maps class IDs to the right names on upload. Getting that mapping wrong does
not error out -- it silently shuffles every class, so the mapping is written
to disk AND printed every run.

IMPORTANT caveat, printed at the end of every run: models/best.pt (as of
Phase 2/3) has zero training examples of signal_red and signal_green. A model
cannot predict a class it has never seen a positive example of. It will not
draw a single signal box, ever, no matter how --conf is tuned. Every
signal_red/signal_green box in this dataset has to be drawn by hand.
"""

import argparse
import glob
import os

import yaml
from ultralytics import YOLO

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# The unified 9-class list from PLAN.md / data/data.yaml, in ID order. This
# is what gets written into the output data.yaml/classes.txt. It is only
# correct as long as --weights was trained on exactly this class list --
# print it loudly so a mismatch is obvious rather than silently corrupting
# every label on upload.
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

NO_SIGNAL_DATA_CLASSES = {"signal_red", "signal_green"}


def find_images(frames_dir):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(frames_dir, "**", f"*{ext}"), recursive=True))
    return sorted(images)


def write_class_mapping(out_dir, class_names):
    """classes.txt (Roboflow's plain-text class-order file) and a data.yaml
    snippet, both in ID order, so the ID -> name mapping is explicit rather
    than implied."""
    os.makedirs(out_dir, exist_ok=True)

    classes_txt = os.path.join(out_dir, "classes.txt")
    with open(classes_txt, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        yaml.safe_dump({"nc": len(class_names), "names": class_names}, f, sort_keys=False)

    print("class ID mapping written to classes.txt / data.yaml (upload this file to Roboflow so IDs land on the right names):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    return classes_txt, data_yaml


def label_path_for(image_path, frames_dir, out_dir):
    """Mirror the image's path under frames_dir into out_dir, .txt extension."""
    rel = os.path.relpath(image_path, frames_dir)
    rel_txt = os.path.splitext(rel)[0] + ".txt"
    return os.path.join(out_dir, rel_txt)


def prelabel_image(model, image_path, label_path, conf, class_names, imgsz):
    """Run detection on one image, write its YOLO label file, return the
    list of class names detected in it (possibly empty)."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    results = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]
    boxes = result.boxes

    detected_names = []
    lines = []
    if boxes is not None:
        img_h, img_w = result.orig_shape
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            detected_names.append(cls_name)

            # YOLO label format wants normalised centre-x, centre-y, w, h,
            # not the pixel xyxy the model gives back.
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # always write the label file, even when empty, so Roboflow (and anyone
    # scanning the output dir) can tell "model looked and found nothing"
    # apart from "model never ran on this frame".
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return detected_names


def main():
    parser = argparse.ArgumentParser(
        description="Pre-label frames with the current model for Roboflow (Phase 4 step 4)"
    )
    parser.add_argument("frames_dir", help="directory of frames to label (searched recursively)")
    parser.add_argument("--weights", default="models/best.pt", help="model weights (default models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold (default 0.3, per PLAN.md)")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help=(
            "inference resolution (default 1280, NOT the training default of 640). Measured on "
            "this footage: a signal head is roughly 20px wide in a 1080p frame. At imgsz=640 the "
            "model finds almost no signals (2 detections on a test clip, best conf 0.52, missing "
            "an obvious red lamp mid-frame). At imgsz=1280 it finds 6, best conf 0.74; at 1920, "
            "0.82. Cost is 91-110ms/frame at 1280 vs 47ms at 640 on CPU -- worth paying here "
            "because this script's whole point is pre-labelling signals well enough to save "
            "labeller time; train.py stays at imgsz=640 since letterboxing every training image "
            "up to 1280 would blow up training time for a gain this script does not need "
            "(training sees the whole dataset repeatedly, not one pass)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write .txt labels (default: next to each image, inside frames_dir)",
    )
    args = parser.parse_args()

    images = find_images(args.frames_dir)
    if not images:
        print(f"no images found under {args.frames_dir}")
        raise SystemExit(1)

    out_dir = args.out_dir if args.out_dir is not None else args.frames_dir

    model = YOLO(args.weights)
    class_names = model.names
    # model.names is normally a dict {0: "car", ...}; turn it into a plain
    # ID-ordered list for both indexing and printing.
    class_names = [class_names[i] for i in sorted(class_names)]

    if class_names != UNIFIED_NAMES:
        print("warning: model's class list does not match the expected unified 9-class list.")
        print(f"  model:    {class_names}")
        print(f"  expected: {UNIFIED_NAMES}")
        print("  writing labels using the MODEL's own order -- double check before uploading.")

    write_class_mapping(out_dir, class_names)

    per_class_totals = {name: 0 for name in class_names}
    zero_detection_count = 0

    print(f"\nrunning {args.weights} at conf={args.conf} imgsz={args.imgsz} over {len(images)} frames...")
    for image_path in images:
        label_path = label_path_for(image_path, args.frames_dir, out_dir)
        detected_names = prelabel_image(model, image_path, label_path, args.conf, class_names, args.imgsz)

        if not detected_names:
            zero_detection_count += 1
        for name in detected_names:
            per_class_totals[name] += 1

    print(f"\nlabelled {len(images)} frames -> {out_dir}")
    print(f"frames with zero detections (need fully manual boxes): {zero_detection_count} / {len(images)}")
    print("\nper-class detection totals:")
    for name in class_names:
        flag = "  (no training data -- see caveat below)" if name in NO_SIGNAL_DATA_CLASSES else ""
        print(f"  {name:<14} {per_class_totals[name]}{flag}")

    print(
        "\nCAVEAT: models/best.pt has no signal_red / signal_green training examples as of "
        "Phase 2/3. It cannot and will not pre-label any traffic signal, at any --conf. "
        "Every signal_red and signal_green box in this dataset must be drawn by hand in "
        "Roboflow -- pre-labelling only saves time on vehicles, person and crosswalk."
    )


if __name__ == "__main__":
    main()
