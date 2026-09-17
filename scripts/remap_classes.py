"""
Phase 1 step 3: remap a downloaded dataset's class IDs onto our unified
9-class list, dropping everything we don't want.

Takes a source dataset directory (as downloaded by download_datasets.py, in
YOLOv8 format: train/valid/test splits, each with images/ and labels/) and a
JSON file mapping {old_id: new_id}. Any old ID not present as a key in the
mapping is dropped entirely.

Safety: this script rewrites label files in place. Running it twice on the
same directory would treat already-new IDs as if they were old IDs again and
silently corrupt the data. A marker file (.remapped) is written after a
successful real run and checked before any further real run. --dry-run never
writes the marker and can be run as many times as you like.
"""

import argparse
import glob
import json
import os

MARKER_NAME = ".remapped"

# our unified 9-class list, in ID order, used only for printing readable
# per-class counts. It has nothing to do with the source dataset's own names.
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


def find_label_files(dataset_dir):
    """All .txt label files under any split's labels/ directory."""
    pattern = os.path.join(dataset_dir, "*", "labels", "*.txt")
    return sorted(glob.glob(pattern))


def find_source_names(dataset_dir):
    """Best-effort: read data.yaml for old-id -> old-name, for printing."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        return {}
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {}


def count_classes(label_files, source_names):
    """Per-old-class-id annotation counts across all label files."""
    counts = {}
    for path in label_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                old_id = int(line.split()[0])
                counts[old_id] = counts.get(old_id, 0) + 1
    return counts


def print_counts(title, counts, name_lookup):
    print(f"\n{title}")
    if not counts:
        print("  (no annotations found)")
        return
    for cls_id in sorted(counts):
        name = name_lookup.get(cls_id, "?")
        print(f"  id {cls_id:>3}  {name:<20}  {counts[cls_id]}")


def remap_label_file(path, mapping):
    """Rewrite one label file, dropping unmapped classes. Returns True if
    the resulting file has at least one annotation left."""
    kept_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            if old_id in mapping:
                new_id = mapping[old_id]
                parts[0] = str(new_id)
                kept_lines.append(" ".join(parts))

    if kept_lines:
        with open(path, "w") as f:
            f.write("\n".join(kept_lines) + "\n")
        return True
    else:
        return False


def image_path_for_label(label_path):
    """labels/foo.txt -> images/foo.<ext>, trying common extensions."""
    images_dir = os.path.join(os.path.dirname(os.path.dirname(label_path)), "images")
    stem = os.path.splitext(os.path.basename(label_path))[0]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Remap a dataset's class IDs to the unified 9-class list")
    parser.add_argument("dataset_dir", help="path to the downloaded dataset (e.g. data/datasets/indian_roads)")
    parser.add_argument("mapping_json", help="path to a JSON file of {old_id: new_id}")
    parser.add_argument("--dry-run", action="store_true", help="print before/after counts without changing anything")
    args = parser.parse_args()

    marker_path = os.path.join(args.dataset_dir, MARKER_NAME)
    if not args.dry_run and os.path.isfile(marker_path):
        print(f"error: {marker_path} exists, meaning this dataset was already remapped.")
        print("running remap again would treat new IDs as old IDs and corrupt the labels.")
        print("re-download the dataset first if you need to redo this.")
        raise SystemExit(1)

    with open(args.mapping_json) as f:
        raw_mapping = json.load(f)
    # JSON keys are always strings; convert both sides to int.
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}

    source_names = find_source_names(args.dataset_dir)
    label_files = find_label_files(args.dataset_dir)
    if not label_files:
        print(f"error: no label .txt files found under {args.dataset_dir}/*/labels/")
        raise SystemExit(1)

    print(f"found {len(label_files)} label files under {args.dataset_dir}")

    before_counts = count_classes(label_files, source_names)
    print_counts("BEFORE (source class ids/names):", before_counts, source_names)

    # what the after-counts would look like, translated through the mapping,
    # for both dry-run and real run.
    after_counts = {}
    for old_id, n in before_counts.items():
        if old_id in mapping:
            new_id = mapping[old_id]
            after_counts[new_id] = after_counts.get(new_id, 0) + n

    unified_lookup = {i: name for i, name in enumerate(UNIFIED_NAMES)}
    print_counts("AFTER (unified class ids/names) — projected:", after_counts, unified_lookup)

    if args.dry_run:
        print("\n--dry-run: no files were modified.")
        return

    removed_images = 0
    for label_path in label_files:
        kept = remap_label_file(label_path, mapping)
        if not kept:
            img_path = image_path_for_label(label_path)
            os.remove(label_path)
            if img_path:
                os.remove(img_path)
                removed_images += 1
            else:
                print(f"warning: no matching image found for empty label {label_path}")

    print(f"\nremoved {removed_images} images whose labels ended up empty")

    with open(marker_path, "w") as f:
        f.write("remapped, do not run remap_classes.py on this directory again\n")

    print(f"wrote marker {marker_path}")
    print("done.")


if __name__ == "__main__":
    main()
