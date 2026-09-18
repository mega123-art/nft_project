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

--- Phase 3: --subsample-sources / --subsample-total ---
The three signal-colour datasets (signals_group_e, signals_detection,
signals_small) add roughly 20,000 images, dwarfing indian_roads +
zebra_crossing. PLAN.md already warned about this exact failure mode for
DriveIndia in Phase 3 ("Full DriveIndia will dominate the crossing/signal
classes"), so the same subsample-before-merge treatment is used here: give
--subsample-sources a list of source directory names to cap, and
--subsample-total the combined image budget across just those sources
(default 1500). Sampling is stratified per source, and within each source
across three buckets -- images with only a signal_red box, only a
signal_green box, or both -- picked round-robin, so a source that happens
to have 10x more red than green frames doesn't quietly make signal_green
the rarer class again after all this effort to give it data. Sources not
named in --subsample-sources are passed through untouched.

--- --data-yaml-out ---
Lets a merge write to data/data_full.yaml or data/data_sub.yaml instead of
data/data.yaml, so the Phase 2 comparison point (data/data.yaml, pointing
at data/datasets/merged/) is never touched by a Phase 3 experiment.
"""

import argparse
import glob
import os
import random
import shutil
from collections import defaultdict

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


def bucket_by_signal(label_path):
    """Classify one label file as 'red', 'green', 'both' or 'neither' based
    on whether it carries a signal_red (7) and/or signal_green (8) box."""
    has_red = has_green = False
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id == 7:
                has_red = True
            elif cls_id == 8:
                has_green = True
    if has_red and has_green:
        return "both"
    if has_red:
        return "red"
    if has_green:
        return "green"
    return "neither"


def stratified_sample(items, n):
    """Pick up to n items from a (image, label, prefix) list, round-robin
    across the red/green/both/neither buckets so no single colour ends up
    starved just because its source happened to shoot more of one colour."""
    buckets = {"red": [], "green": [], "both": [], "neither": []}
    for item in items:
        buckets[bucket_by_signal(item[1])].append(item)
    for bucket in buckets.values():
        random.shuffle(bucket)

    order = ["red", "green", "both", "neither"]
    positions = {b: 0 for b in order}
    chosen = []
    while len(chosen) < n:
        progressed = False
        for b in order:
            if len(chosen) >= n:
                break
            if positions[b] < len(buckets[b]):
                chosen.append(buckets[b][positions[b]])
                positions[b] += 1
                progressed = True
        if not progressed:
            break  # every bucket exhausted, fewer than n items available
    return chosen


def subsample_by_source(all_items, subsample_prefixes, target_total):
    """Cap the combined image count of the named source prefixes at
    target_total, stratified per source and per signal colour (see
    stratified_sample). Sources not in subsample_prefixes pass through
    untouched. Returns the new combined item list."""
    by_prefix = defaultdict(list)
    passthrough = []
    for item in all_items:
        _, _, prefix = item
        if prefix in subsample_prefixes:
            by_prefix[prefix].append(item)
        else:
            passthrough.append(item)

    missing = [p for p in subsample_prefixes if p not in by_prefix]
    if missing:
        print(f"warning: --subsample-sources named {missing}, but no items from those sources were found")

    present = [p for p in subsample_prefixes if p in by_prefix]
    n_sources = len(present)
    if n_sources == 0:
        return all_items

    base_quota = target_total // n_sources
    remainder = target_total - base_quota * n_sources
    quotas = {p: base_quota for p in present}
    for p in present[:remainder]:  # deterministic, no randomness needed for a +1
        quotas[p] += 1

    selected = {}
    for p in present:
        take_n = min(quotas[p], len(by_prefix[p]))
        selected[p] = stratified_sample(by_prefix[p], take_n)

    # top up any shortfall (a source with fewer images than its quota) from
    # whichever source has the most spare capacity left, largest first.
    shortfall = target_total - sum(len(v) for v in selected.values())
    if shortfall > 0:
        chosen_ids = {id(item) for items in selected.values() for item in items}
        by_spare_capacity = sorted(present, key=lambda p: len(by_prefix[p]) - len(selected[p]), reverse=True)
        for p in by_spare_capacity:
            if shortfall <= 0:
                break
            remaining_pool = [item for item in by_prefix[p] if id(item) not in chosen_ids]
            take_n = min(shortfall, len(remaining_pool))
            if take_n <= 0:
                continue
            extra = stratified_sample(remaining_pool, take_n)
            selected[p].extend(extra)
            chosen_ids.update(id(item) for item in extra)
            shortfall -= take_n

    print(f"\nsubsampling {present} down to ~{target_total} images total:")
    for p in present:
        counts = count_classes([label_path for _, label_path, _ in selected[p]])
        print(
            f"  {p}: {len(by_prefix[p])} available -> {len(selected[p])} selected "
            f"(signal_red {counts.get(7, 0)}, signal_green {counts.get(8, 0)})"
        )
    if shortfall > 0:
        print(f"  warning: could not reach {target_total} images, {shortfall} short (sources ran out)")

    combined = list(passthrough)
    for p in present:
        combined.extend(selected[p])
    return combined


def main():
    parser = argparse.ArgumentParser(description="Merge remapped datasets into one 85/15 train/val split")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="remapped dataset directories to merge")
    parser.add_argument("--out-dir", default="data/datasets/merged", help="output merged dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="fraction of images held out for val")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument(
        "--data-yaml-out",
        default="data/data.yaml",
        help="where to write the unified data.yaml (default data/data.yaml -- pass a different path for an "
        "experimental merge so the Phase 2 comparison point is never overwritten)",
    )
    parser.add_argument(
        "--subsample-sources",
        nargs="+",
        default=None,
        metavar="SOURCE_DIR_NAME",
        help="source directory basenames (as they appear in --sources) to cap at --subsample-total combined "
        "images, stratified by signal_red/signal_green. Default: no subsampling.",
    )
    parser.add_argument(
        "--subsample-total",
        type=int,
        default=1500,
        help="combined image budget across all --subsample-sources (default 1500)",
    )
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

    # seed before ANY randomness -- both the subsampling below and the
    # train/val shuffle further down draw from this same seeded stream, so
    # the whole merge is reproducible from --seed alone.
    random.seed(args.seed)

    if args.subsample_sources:
        all_items = subsample_by_source(all_items, args.subsample_sources, args.subsample_total)

    before_counts = count_classes([label_path for _, label_path, _ in all_items])
    print_histogram("class histogram BEFORE split (all pooled sources):", before_counts)

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

    data_yaml_path = args.data_yaml_out
    os.makedirs(os.path.dirname(data_yaml_path) or ".", exist_ok=True)
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
