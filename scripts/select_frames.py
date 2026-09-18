"""
Phase 4 step 5 (new, not numbered in PLAN.md but required before Roboflow
upload): pick the ~800-1200 frames a human labeller actually has time for
out of everything scripts/extract_frames.py + scripts/prelabel.py produced.

Why this exists: extract_frames.py is deliberately generous per clip (see
the per-clip interval choices recorded when it was run for Phase 4 -- short
signal/crossing clips are sampled tightly because they are the only source
of signal_red/signal_green and crosswalk data, long walking-tour clips are
sampled loosely because 0.5s on 30 minutes of pavement would be 4300 near-
identical frames). Even after that judgement call, the resulting pool is
bigger than PLAN.md's Phase 4 step 5 budget (800-1200 corrected frames).
Handing the team the whole pool means most of their time goes to frames
that teach the model nothing it doesn't already know (another car at 0.93
transfer confidence) instead of the frames that matter (a signal, a
crosswalk, an autorickshaw).

Ranking, in priority order (highest value first):
  0. signal_red / signal_green present  -- these classes have ZERO Indian
     training data before Phase 4 and the model cannot pre-label them at
     any confidence. Every one of these frames is irreplaceable.
  1. crosswalk present                  -- only 417 crosswalk annotations
     exist across all datasets combined; still comparatively rare.
  2. autorickshaw / bus present, or an unusually crowded frame (>=6 person
     boxes) -- underrepresented classes/scenarios the public data (dashcam
     footage, mostly non-Indian) does not cover well.
  3. any other frame with at least one detection -- normal traffic mix,
     still worth correcting but not scarce.
  4. "boring" frames -- zero detections, or only 1-2 car/truck boxes and
     nothing else. The model already gets car at 0.93 transfer confidence
     (see report/phase4_transfer_check); more of these teaches it little.

Tier 4 is never dropped to zero on purpose. A dataset made of only
"interesting" frames biases the model toward always-something-worth-
detecting scenes and inflates apparent accuracy, because the model never
has to learn what an empty/ordinary road looks like. --boring-fraction
(default 0.15) reserves a slice of the budget for tier 4 regardless of how
much tier 0-3 material is available.

Two more rules keep the selection useful rather than just "top N by score":
  - --max-frac-per-video caps how much of the budget a single clip can
    consume (default 0.35), except tier 0 (signal) frames, which are always
    kept -- there are only a handful of them across all footage and losing
    even one to a per-video cap would be a bad trade for a "diversity" rule
    that exists to stop one clip dominating on volume, not on scarcity.
  - A perceptual (average) hash de-duplicates near-identical consecutive
    frames of the same video (e.g. two frames 0.1s apart of a stationary
    signal with nothing else moving) so the budget is not spent on
    near-copies. Tier 0 frames skip this too, for the same scarcity reason
    -- a signal changing state between two close frames is exactly the kind
    of near-duplicate-looking pair we want BOTH copies of.

Output: images + their .txt labels copied into --out-dir (default
data/frames_selected/), plus classes.txt/data.yaml (same convention as
prelabel.py, so Roboflow gets the same explicit ID->name mapping) and a
manifest.csv recording, per selected frame, which tier/reason it was picked
for -- so the team (and the report) can see the sampling method, not just
the result.
"""

import argparse
import csv
import glob
import os
import random
import shutil

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Same unified 9-class list as prelabel.py / pseudo_label.py / PLAN.md, in ID
# order. Kept as a literal copy (not imported) because these scripts are
# meant to be run independently against whatever labels are on disk.
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
CAR_ID, BUS_ID, TRUCK_ID, MOTO_ID, AUTO_ID, PERSON_ID, CROSSWALK_ID, SIGNAL_RED_ID, SIGNAL_GREEN_ID = range(9)

CROWD_THRESHOLD = 6  # person boxes at/above this count counts as "unusually crowded" (rule 2)
BORING_MAX_BOXES = 2  # tier-4 cutoff: at most this many boxes, and only from BORING_CLASSES
BORING_CLASSES = {CAR_ID, TRUCK_ID}

TIER_LABELS = {
    0: "signal_red/signal_green present -- zero Indian training data, irreplaceable",
    1: "crosswalk present -- only 417 crosswalk annotations exist across all datasets",
    2: "autorickshaw/bus present or unusually crowded (person >= 6)",
    3: "ordinary frame with at least one detection",
    4: "boring: zero, or only 1-2 car/truck boxes and nothing else",
}


def find_pairs(frames_dir):
    """(image_path, label_path, video) for every image under frames_dir that
    has a matching .txt label file. video is the immediate parent dir name,
    matching extract_frames.py's per-video output layout."""
    pairs = []
    for ext in IMAGE_EXTENSIONS:
        for image_path in glob.glob(os.path.join(frames_dir, "**", f"*{ext}"), recursive=True):
            label_path = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.isfile(label_path):
                continue
            video = os.path.basename(os.path.dirname(image_path))
            pairs.append((image_path, label_path, video))
    return sorted(pairs)


def read_classes(label_path):
    """List of class ids in this frame's label file (may repeat, may be
    empty)."""
    ids = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(line.split()[0]))
    return ids


def classify(class_ids):
    """Return (tier, reason) for one frame given its list of class ids."""
    counts = {i: class_ids.count(i) for i in set(class_ids)}

    if SIGNAL_RED_ID in counts or SIGNAL_GREEN_ID in counts:
        colours = []
        if SIGNAL_RED_ID in counts:
            colours.append(f"signal_red x{counts[SIGNAL_RED_ID]}")
        if SIGNAL_GREEN_ID in counts:
            colours.append(f"signal_green x{counts[SIGNAL_GREEN_ID]}")
        return 0, "tier0: " + ", ".join(colours)

    if CROSSWALK_ID in counts:
        return 1, f"tier1: crosswalk x{counts[CROSSWALK_ID]}"

    if AUTO_ID in counts or BUS_ID in counts:
        parts = []
        if AUTO_ID in counts:
            parts.append(f"autorickshaw x{counts[AUTO_ID]}")
        if BUS_ID in counts:
            parts.append(f"bus x{counts[BUS_ID]}")
        return 2, "tier2: " + ", ".join(parts)

    if counts.get(PERSON_ID, 0) >= CROWD_THRESHOLD:
        return 2, f"tier2: crowded, person x{counts[PERSON_ID]}"

    if not class_ids:
        return 4, "tier4: boring, zero detections"

    non_boring = [c for c in class_ids if c not in BORING_CLASSES]
    if not non_boring and len(class_ids) <= BORING_MAX_BOXES:
        return 4, f"tier4: boring, only {len(class_ids)} car/truck box(es)"

    return 3, "tier3: ordinary frame with detections (" + ", ".join(
        f"{UNIFIED_NAMES[i]} x{counts[i]}" for i in sorted(counts)
    ) + ")"


def average_hash(image_path, hash_size=8):
    """Cheap perceptual hash for near-duplicate detection: greyscale,
    shrink to hash_size x hash_size, threshold against the mean. Two frames
    a fraction of a second apart of a mostly-static scene hash identically
    or nearly so; a scene that has actually changed (new vehicle entering,
    signal changing) does not."""
    img = Image.open(image_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32).flatten()
    avg = pixels.mean()
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return int(bits)


def hamming(a, b):
    return bin(a ^ b).count("1")


def is_near_duplicate(candidate_hash, selected_hashes, threshold):
    return any(hamming(candidate_hash, h) <= threshold for h in selected_hashes)


def write_class_mapping(out_dir):
    """Same convention as prelabel.py: classes.txt + data.yaml in ID order,
    written and printed loudly so the ID->name mapping on this subset can
    never silently drift from the mapping used to generate it."""
    import yaml

    classes_txt = os.path.join(out_dir, "classes.txt")
    with open(classes_txt, "w") as f:
        for name in UNIFIED_NAMES:
            f.write(name + "\n")

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        yaml.safe_dump({"nc": len(UNIFIED_NAMES), "names": UNIFIED_NAMES}, f, sort_keys=False)

    print("class ID mapping written to classes.txt / data.yaml (same IDs as prelabel.py -- upload this file):")
    for i, name in enumerate(UNIFIED_NAMES):
        print(f"  {i}: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank and select the highest-value pre-labelled frames for the Roboflow labelling budget (Phase 4)"
    )
    parser.add_argument("frames_dir", nargs="?", default="data/frames", help="root of extracted+pre-labelled frames (default data/frames)")
    parser.add_argument("--out-dir", default="data/frames_selected", help="where to copy the selected images+labels (default data/frames_selected)")
    parser.add_argument("--budget", type=int, default=1000, help="total number of frames to select (default 1000, per PLAN.md's 800-1200 target)")
    parser.add_argument("--max-frac-per-video", type=float, default=0.35, help="cap on the fraction of the budget any single video may contribute, except tier-0 signal frames which are exempt (default 0.35)")
    parser.add_argument("--boring-fraction", type=float, default=0.15, help="fraction of the budget deliberately reserved for tier-4 'boring' frames, so the dataset is not only interesting frames (default 0.15)")
    parser.add_argument("--dup-hash-threshold", type=int, default=4, help="max hamming distance (of 64 bits) for two frames of the same video to count as near-duplicates; tier-0 frames are exempt (default 4)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for boring-frame sampling and tie-breaking (default 0)")
    args = parser.parse_args()

    pairs = find_pairs(args.frames_dir)
    if not pairs:
        print(f"no (image, label) pairs found under {args.frames_dir}")
        raise SystemExit(1)

    rng = random.Random(args.seed)

    # Classify every frame once.
    classified = []  # (tier, reason, image_path, label_path, video)
    for image_path, label_path, video in pairs:
        class_ids = read_classes(label_path)
        tier, reason = classify(class_ids)
        classified.append((tier, reason, image_path, label_path, video))

    tier_counts = {t: 0 for t in range(5)}
    for tier, *_ in classified:
        tier_counts[tier] += 1
    print(f"found {len(classified)} pre-labelled frames under {args.frames_dir}")
    for t in range(5):
        print(f"  tier {t} ({TIER_LABELS[t]}): {tier_counts[t]} frames")

    videos = sorted({v for *_, v in classified})
    video_cap = max(1, round(args.budget * args.max_frac_per_video))
    print(f"\nper-video cap (except tier-0 signal frames, which are always kept): {video_cap} frames / video")

    boring_budget = round(args.budget * args.boring_fraction)
    main_budget = args.budget - boring_budget
    print(f"budget split: {main_budget} frames from tiers 0-3 (ranked), {boring_budget} from tier 4 (sampled, to keep 'ordinary' data in the mix)")

    # Rank tiers 0-3 by tier, then shuffle within tier so ties don't always
    # favour whichever video happens to sort first alphabetically.
    ranked = [c for c in classified if c[0] <= 3]
    ranked.sort(key=lambda c: c[0])
    # stable shuffle within each tier
    by_tier = {}
    for c in ranked:
        by_tier.setdefault(c[0], []).append(c)
    for t in by_tier:
        rng.shuffle(by_tier[t])
    ranked = [c for t in sorted(by_tier) for c in by_tier[t]]

    boring_pool = [c for c in classified if c[0] == 4]
    rng.shuffle(boring_pool)

    selected = []
    video_selected_count = {v: 0 for v in videos}
    video_hashes = {v: [] for v in videos}

    def try_select(entry, respect_cap, respect_dedup):
        tier, reason, image_path, label_path, video = entry
        if respect_cap and video_selected_count[video] >= video_cap:
            return False
        if respect_dedup:
            h = average_hash(image_path)
            if is_near_duplicate(h, video_hashes[video], args.dup_hash_threshold):
                return False
        else:
            h = average_hash(image_path)
        selected.append(entry)
        video_selected_count[video] += 1
        video_hashes[video].append(h)
        return True

    # Pass 1: tiers 0-3, up to main_budget. Tier 0 is exempt from both the
    # per-video cap and de-duplication -- see module docstring.
    for entry in ranked:
        if len(selected) >= main_budget:
            break
        tier = entry[0]
        if tier == 0:
            try_select(entry, respect_cap=False, respect_dedup=False)
        else:
            try_select(entry, respect_cap=True, respect_dedup=True)

    # Backfill from tier 4 if tiers 0-3 didn't fill main_budget (small
    # footage pool case).
    leftover_boring = list(boring_pool)
    i = 0
    while len(selected) < main_budget and i < len(leftover_boring):
        try_select(leftover_boring[i], respect_cap=True, respect_dedup=True)
        i += 1
    leftover_boring = leftover_boring[i:]

    # Pass 2: tier 4 boring quota.
    target_total = len(selected) + boring_budget
    j = 0
    while len(selected) < target_total and j < len(leftover_boring):
        try_select(leftover_boring[j], respect_cap=True, respect_dedup=True)
        j += 1

    # If tier 4 ran out (small footage pool), backfill remaining budget
    # from whatever tiers 0-3 material is left over (dedup/cap still
    # enforced, tier 0 already exhausted by pass 1).
    remaining_ranked = [c for c in ranked if c not in selected]
    k = 0
    while len(selected) < args.budget and k < len(remaining_ranked):
        entry = remaining_ranked[k]
        try_select(entry, respect_cap=(entry[0] != 0), respect_dedup=(entry[0] != 0))
        k += 1

    if len(selected) < args.budget:
        print(f"\nnote: only {len(selected)} frames available/selectable after caps and de-dup, below the requested budget of {args.budget}.")

    # Write output.
    images_out = os.path.join(args.out_dir, "images")
    labels_out = os.path.join(args.out_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    write_class_mapping(args.out_dir)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "video", "tier", "reason", "num_boxes", "classes_present"])
        for tier, reason, image_path, label_path, video in selected:
            fname = os.path.basename(image_path)
            shutil.copy2(image_path, os.path.join(images_out, fname))
            shutil.copy2(label_path, os.path.join(labels_out, os.path.splitext(fname)[0] + ".txt"))
            class_ids = read_classes(label_path)
            classes_present = ";".join(sorted({UNIFIED_NAMES[c] for c in class_ids})) or "(none)"
            writer.writerow([fname, video, tier, reason, len(class_ids), classes_present])

    print(f"\nselected {len(selected)} frames -> {args.out_dir}")
    print(f"manifest written to {manifest_path}")

    final_tier_counts = {t: 0 for t in range(5)}
    final_video_counts = {v: 0 for v in videos}
    for tier, reason, image_path, label_path, video in selected:
        final_tier_counts[tier] += 1
        final_video_counts[video] += 1
    print("\nselected frames by tier:")
    for t in range(5):
        print(f"  tier {t}: {final_tier_counts[t]}")
    print("\nselected frames by video:")
    for v in videos:
        print(f"  {v}: {final_video_counts[v]}")


if __name__ == "__main__":
    main()
