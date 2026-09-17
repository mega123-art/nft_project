"""
Phase 1 step 4: merge the remapped datasets into one YOLO dataset with an
85/15 train/val split, and write data/data.yaml describing it.

Input: one or more dataset directories that have already been through
remap_classes.py (so their label files already use our unified 0-8 class
ids, mixed across whatever splits the original download had — train/valid/
test all get pooled and re-split here).

Output:
    data/datasets/merged/train/images, train/labels
    data/datasets/merged/val/images,   val/labels
    data/data.yaml

Filenames from different source datasets can collide (e.g. both call an
image "IMG_001.jpg"), so every copied file is prefixed with its source
dataset's directory name.
"""

import argparse
import glob
import os
import random
import shutil

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

DEFAULT_SOURCES = [
    "data/datasets/indian_roads",
    "data/datasets/zebra_crossing",
]


def collect_pairs(dataset_dir):
    """All (image_path, label_path) pairs across every split in this dataset."""
    pairs = []
    label_paths = sorted(glob.glob(os.path.join(dataset_dir, "*", "labels", "*.txt")))
    for label_path in label_paths:
        images_dir = os.path.join(os.path.dirname(os.path.dirname(label_path)), "images")
        stem = os.path.splitext(os.path.basename(label_path))[0]
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.isfile(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"warning: no image found for label {label_path}, skipping")
            continue
        pairs.append((image_path, label_path))
    return pairs


def count_classes(label_paths):
    counts = {}
    for path in label_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def print_histogram(title, counts):
    print(f"\n{title}")
    total = sum(counts.values())
    for i, name in enumerate(UNIFIED_NAMES):
        print(f"  {i}  {name:<14}  {counts.get(i, 0)}")
    print(f"  total annotations: {total}")


def main():
    parser = argparse.ArgumentParser(description="Merge remapped datasets into one 85/15 train/val split")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="remapped dataset directories to merge")
    parser.add_argument("--out-dir", default="data/datasets/merged", help="output merged dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="fraction of images held out for val")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    args = parser.parse_args()

    for src in args.sources:
        marker = os.path.join(src, ".remapped")
        if not os.path.isfile(marker):
            print(f"error: {src} has no .remapped marker -- run remap_classes.py on it first")
            raise SystemExit(1)

    # gather all (image, label, source_prefix) triples across all sources
    all_items = []
    for src in args.sources:
        prefix = os.path.basename(os.path.normpath(src))
        pairs = collect_pairs(src)
        print(f"{src}: {len(pairs)} image/label pairs")
        for image_path, label_path in pairs:
            all_items.append((image_path, label_path, prefix))

    if not all_items:
        print("error: no image/label pairs found in any source dataset")
        raise SystemExit(1)

    before_counts = count_classes([label_path for _, label_path, _ in all_items])
    print_histogram("class histogram BEFORE split (all pooled sources):", before_counts)

    random.seed(args.seed)
    random.shuffle(all_items)

    n_val = int(round(len(all_items) * args.val_fraction))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    # clean start so reruns don't leave stale files behind
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    for split_name, items in (("train", train_items), ("val", val_items)):
        images_dir = os.path.join(args.out_dir, split_name, "images")
        labels_dir = os.path.join(args.out_dir, split_name, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        for image_path, label_path, prefix in items:
            new_stem = f"{prefix}_{os.path.splitext(os.path.basename(image_path))[0]}"
            image_ext = os.path.splitext(image_path)[1]
            shutil.copy2(image_path, os.path.join(images_dir, new_stem + image_ext))
            shutil.copy2(label_path, os.path.join(labels_dir, new_stem + ".txt"))

    print(f"\ntrain: {len(train_items)} images -> {args.out_dir}/train")
    print(f"val:   {len(val_items)} images -> {args.out_dir}/val")

    train_label_files = glob.glob(os.path.join(args.out_dir, "train", "labels", "*.txt"))
    val_label_files = glob.glob(os.path.join(args.out_dir, "val", "labels", "*.txt"))
    print_histogram("class histogram TRAIN:", count_classes(train_label_files))
    print_histogram("class histogram VAL:", count_classes(val_label_files))

    data_yaml_path = "data/data.yaml"
    abs_out_dir = os.path.abspath(args.out_dir)
    with open(data_yaml_path, "w") as f:
        f.write(f"train: {abs_out_dir}/train/images\n")
        f.write(f"val: {abs_out_dir}/val/images\n")
        f.write(f"nc: {len(UNIFIED_NAMES)}\n")
        f.write("names:\n")
        for name in UNIFIED_NAMES:
            f.write(f"  - {name}\n")
    print(f"\nwrote {data_yaml_path}")


if __name__ == "__main__":
    main()
