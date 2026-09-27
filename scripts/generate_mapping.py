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

# signal_countdown (11) has no source names below and is never produced by
# this script's mapping -- see PLAN.md's class table and data/LABELLING.md.
# Listed anyway so UNIFIED_ID/UNIFIED_NAMES stay identical across every
# copy of this list in the repo.
#
# --- signal_red/signal_green split into ped_*/veh_* (post-review fix) ---
# A reviewer found a false-safe path: the old signal_red/signal_green pair
# conflated PEDESTRIAN signals (walking-man icon, means "you may walk") with
# VEHICLE traffic lights (means "cars may go"), and src/fsm.py's SAFE rule
# treated ANY signal_green the same way. A green VEHICLE light is not
# permission for a pedestrian to cross -- it is the opposite, it means
# traffic has right of way. See data/LABELLING.md section 2 for the full
# writeup of why the old shared-class rule was wrong.
#
# The split is mechanically clean because of how the source datasets happen
# to name things: ono-gedd7/pedestrian-traffic-light-puf4a (the only
# pedestrian-signal dataset in this project) uses bare lowercase "green"/
# "red", while every vehicle-signal dataset (group-e, traffic-light-
# detection-l869b, traffic-si0cm, fyp-wrdsh) uses capitalised or prefixed
# names (Green, GreenLeft, Red Light, Traffic_light_green, ...). Verified by
# reading every data.yaml under data/datasets/ before relying on this: none
# of indian_roads/signals_detection/signals_small/zebra_crossing carry a
# bare lowercase "green" or "red", so this split cannot silently move a
# vehicle-light box into the pedestrian classes or vice versa.
UNIFIED_NAMES = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "autorickshaw",
    "person",
    "crosswalk",
    "ped_signal_red",
    "ped_signal_green",
    "veh_signal_red",
    "veh_signal_green",
    "signal_countdown",
    # Colourless signal boxes (a signal head whose lit lamp cannot be read).
    # Asserts only "a signal exists here", never a colour. src/fsm.py never
    # reads it; it exists so these boxes are not left unlabelled and thus
    # learned as background. See the mapping entries below for the full
    # reasoning.
    "signal_unknown",
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

    # --- Phase 3: signal-colour datasets (group-e, traffic-light-detection,
    # traffic-si0cm) --- these are ALL vehicle traffic lights (dashcam-style
    # exports, shot from inside a car -- see download_signal_datasets.py's
    # docstring), so every Red*/Green* variant maps to the veh_signal_*
    # classes, never the ped_signal_* ones. All Red* variants (including
    # amber/yellow) map to veh_signal_red, all Green* variants map to
    # veh_signal_green. "off" (an unlit lamp) is deliberately NOT in this
    # table at all, so it is dropped by the same "unmapped name -> dropped"
    # path as everything else -- see data/LABELLING.md section 2: an unlit
    # signal gives no evidence and must not be labelled a colour.
    "Red": "veh_signal_red",
    "RedLeft": "veh_signal_red",
    "RedRight": "veh_signal_red",
    "RedStraight": "veh_signal_red",
    "RedStraightLeft": "veh_signal_red",
    "Red Light": "veh_signal_red",
    "Red-Light": "veh_signal_red",  # defensive: not the real export name (see download_signal_datasets.py),
                                     # kept in case a future re-export uses it.
    # Amber/Yellow -> veh_signal_red, NOT a third class and NOT *_green.
    # This is deliberate, not a copy-paste mistake: data/LABELLING.md
    # section 2 rules that amber means "traffic may still be moving" and
    # must be treated as the conservative (red) case, because a false SAFE
    # is the one failure mode this whole project cannot tolerate. That
    # reasoning is unchanged by the ped/veh split -- amber is a vehicle-
    # light phase, so it stays on the veh_signal_red side.
    "Yellow": "veh_signal_red",
    "Yellow Light": "veh_signal_red",
    "Yellow-Light": "veh_signal_red",  # defensive, see Red-Light note above
    "Green": "veh_signal_green",
    "GreenLeft": "veh_signal_green",
    "GreenRight": "veh_signal_green",
    "GreenStraight": "veh_signal_green",
    "GreenStraightLeft": "veh_signal_green",
    "GreenStraightRight": "veh_signal_green",
    "Green Light": "veh_signal_green",
    "Green-Light": "veh_signal_green",  # defensive, see Red-Light note above

    # --- Phase 4: pedestrian-signal round (ono-gedd7, fyp-wrdsh) ---
    # ono-gedd7/pedestrian-traffic-light-puf4a's real data.yaml (v1) uses
    # bare lowercase "green"/"red", and it is the ONLY dataset in this
    # project that actually shows a pedestrian walking-man signal -- so
    # these two keys are the ONLY source names that map to the ped_signal_*
    # classes. Checked against every data.yaml already downloaded into
    # data/datasets/ (indian_roads, signals_detection, signals_small,
    # zebra_crossing) plus the group-e/traffic-si0cm class lists documented
    # in download_signal_datasets.py's docstring: none of them contain a
    # bare lowercase "green" or "red", only capitalised variants (Green,
    # GreenLeft, Red Light, ...), so adding these two keys cannot silently
    # re-map any class already in this table. Still, this was only checked
    # against what's on disk right now -- re-verify against signals_group_e's
    # actual data.yaml once it's downloaded, in case a future export ever
    # introduces a lowercase name there (if it does, and it is genuinely a
    # vehicle light, it must map to veh_signal_*, NOT here).
    "green": "ped_signal_green",
    "red": "ped_signal_red",
    # fyp-wrdsh/road-signs-and-traffic-lights-dataset's Traffic_light_*
    # classes are vehicle signals (see download_signal_datasets.py's
    # docstring: fyp-wrdsh is described as adding "more vehicle
    # signal_red/signal_green"), so these map to the veh_signal_* side,
    # same pattern as group-e/traffic-si0cm above.
    "Traffic_light_green": "veh_signal_green",
    "Traffic_light_red": "veh_signal_red",
    # fyp-wrdsh also already labels car/person/motorcycle directly (lowercase,
    # COCO-style names) -- "person" is already covered by the existing
    # lowercase entry above, "car" and "motorcycle" are new lowercase keys.
    "car": "car",
    "motorcycle": "motorcycle",
    # ono-gedd7's four colourless signal classes. None of them says which
    # lamp is lit, so data/LABELLING.md section 2 forbids giving them a
    # colour -- the same rule that drops indian_roads' "Traffic Signal".
    #
    # They are NOT dropped, though, and that is a deliberate change. In
    # ono-gedd7 they are 1478 boxes (traffic_light 522, "pedestrian Traffic
    # Light" 761, signal-light 164, trafficlight 31) sitting in the SAME
    # images that carry our ped_signal_green/ped_signal_red labels. Dropping
    # a box does not remove the object from the image, it just leaves it
    # unlabelled -- and YOLO treats unlabelled pixels as background, so
    # dropping them would actively teach the model that signal-shaped
    # objects are background. That is the partial-labelling poisoning that
    # held the person class down to 0.576 earlier in this project.
    #
    # signal_unknown absorbs them instead: the boxes stop being background,
    # while asserting nothing about colour. src/fsm.py never reads this
    # class, so it cannot influence a crossing decision in either
    # direction -- it exists purely to protect ped_signal_green, the one
    # class that can license SAFE and the one with the least data (1054).
    "pedestrian Traffic Light": "signal_unknown",
    "traffic_light": "signal_unknown",
    "signal-light": "signal_unknown",
    "trafficlight": "signal_unknown",
    "Traffic Signal": "signal_unknown",  # indian_roads' colourless class
    # "off" (an unlit lamp) stays dropped: an unlit signal is not a signal
    # the model should learn to find, and LABELLING.md section 2 rules it
    # out explicitly.
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
# (Historical note, now stale: this comment used to say signal_red/
# signal_green had no source names at all. That was true before Phase 3/4
# added the signal-colour datasets above. Now: veh_signal_red/veh_signal_green
# get plenty of public data from group-e/traffic-light-detection/traffic-
# si0cm/fyp-wrdsh (all vehicle dashcam shots); ped_signal_red/ped_signal_green
# get public data ONLY from ono-gedd7 (the one pedestrian-signal dataset).
# Our own Phase 4+ footage is still what fixes real-world coverage for both.)
#
# fyp-wrdsh's own ~20 road-sign classes (speed limits, no-entry, bends, etc.)
# are likewise dropped on purpose here by simply not being in the table --
# they are not in our 10-class list and were never claimed to be.


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
