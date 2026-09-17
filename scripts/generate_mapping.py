"""
Phase 1 helper: build the {old_id: new_id} JSON that remap_classes.py needs,
by reading a dataset's real data.yaml and matching class NAMES (not their
order/index, which can differ between exports) against our agreed mapping.

This is a small standalone script, not a library, so it is obvious exactly
which names go where. The name table below is the one agreed for this
project (see the task instructions / PLAN.md "Unified class list").

Usage:
    .venv/bin/python scripts/generate_mapping.py data/datasets/indian_roads
    .venv/bin/python scripts/generate_mapping.py data/datasets/zebra_crossing

Writes <dataset_dir>/class_mapping.json and also prints it.
"""

import argparse
import json
import os

import yaml

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
UNIFIED_ID = {name: i for i, name in enumerate(UNIFIED_NAMES)}

# every source-dataset name we expect to see, mapped to its unified name.
# case-sensitive on purpose: the actual exports are inconsistent about case
# and we want mismatches to show up as "missing", not silently merge wrong
# things.
NAME_TO_UNIFIED = {
    "Car": "car",
    "Ambulance": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Tempo": "truck",
    "Tractor": "truck",
    "MotorBike": "motorcycle",
    "Bike": "motorcycle",
    "Cycle": "motorcycle",
    "Autorickshaw": "autorickshaw",
    "Rikshaw": "autorickshaw",
    "person": "person",
    "Person": "person",
    "Zebra Crossing": "crosswalk",
    "Zebra-Crossing": "crosswalk",
}
# Everything else in the Indian dataset's real (v2) 48-class list is dropped
# on purpose: Traffic Signal (no red/green colour info, so it's useless for
# our signal_red/signal_green classes and would just be noise), the
# Traffic POlice/Traffic Police and Bus Stop/Bus stop and Lamp Post/Lamo
# Post duplicate-capitalisation junk, Cattle/Camel/Goat/Horse/Dog, Building/
# Wall/Gate/Bridge/Overbridge/Footpath/Road Divider/Electricity Pole/Tree/
# Vegetation/Petrol Pump/Tyre Works/Crane/Manhole/Digital Display/Board/
# Sign Board/Traffic Sign Board/Flag/Barricade/Cart, and the junk class
# literally named 81 W's (a bad label in the source project — see the
# printed annotation count below, it should be near zero or this dataset's
# label quality is worse than expected).
# signal_red and signal_green intentionally have no source names: nothing in
# the public datasets carries the red/green distinction. They get zero
# public data by design (Phase 4 fixes this with our own footage).


def load_source_names(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        print(f"error: no data.yaml found at {yaml_path}")
        raise SystemExit(1)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def main():
    parser = argparse.ArgumentParser(description="Generate a {old_id: new_id} mapping from a dataset's data.yaml")
    parser.add_argument("dataset_dir", help="path to the downloaded dataset")
    args = parser.parse_args()

    source_names = load_source_names(args.dataset_dir)
    print(f"actual classes in {args.dataset_dir}/data.yaml:")
    for old_id in sorted(source_names):
        print(f"  {old_id}: {source_names[old_id]!r}")

    mapping = {}
    dropped = []
    for old_id, name in source_names.items():
        if name in NAME_TO_UNIFIED:
            unified_name = NAME_TO_UNIFIED[name]
            mapping[old_id] = UNIFIED_ID[unified_name]
        else:
            dropped.append(name)

    if dropped:
        print(f"\ndropping {len(dropped)} class(es) not in our unified list: {dropped}")

    # loudly warn about expected names (from our full agreed table) that are
    # simply not present anywhere in this dataset's data.yaml. This is the
    # sanity check the task explicitly asked for: exports drift from what we
    # assumed, and silent absence is exactly the failure mode to catch.
    present_names = set(source_names.values())
    missing_expected = [n for n in NAME_TO_UNIFIED if n not in present_names]
    if missing_expected:
        print(f"\nWARNING: {len(missing_expected)} expected name(s) from our mapping table are NOT in this data.yaml:")
        print(f"  {missing_expected}")
        print("  (this just means this particular dataset does not carry those classes -- expected for a")
        print("   single-purpose export, but check it matches what you thought you were downloading)")

    print("\nresulting {old_id: new_id} mapping:")
    for old_id in sorted(mapping):
        print(f"  {old_id} ({source_names[old_id]!r}) -> {mapping[old_id]} ({UNIFIED_NAMES[mapping[old_id]]!r})")

    out_path = os.path.join(args.dataset_dir, "class_mapping.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
