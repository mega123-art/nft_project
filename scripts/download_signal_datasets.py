"""
Phase 3 step 1: download the three public Roboflow traffic-light datasets
used to give signal_red/signal_green their first training data.

- group-e/traffic-lights-lxcaj                              (multi-state traffic lights)
- traffic-light-detection-975ge/traffic-light-detection-l869b (Red/Yellow/Green lights)
- traffic-si0cm/traffic-light-gxodz                          (small Red/Yellow/Green set)

This is a sibling to download_datasets.py rather than an extension of it:
download_datasets.py's docstring and defaults are scoped to "the two public
Roboflow datasets we need" for Phase 1 (indian_roads, zebra_crossing), and
its --indian-version/--zebra-version flags are dataset-specific by design.
Bolting three more unrelated datasets onto that file would mean either
renaming its whole interface or growing a pile of unrelated per-dataset
flags on one already-specific script. Keeping Phase 3 sources in their own
script with the same structure (WORKSPACE_PROJECT table, dir_is_nonempty,
download_one, print_classes) is easier to read and easier to blow away
independently if the signal-data experiment doesn't pan out.

All three are pulled in YOLOv8 export format. The Roboflow API key is never
written to disk: it must be set in the environment as ROBOFLOW_API_KEY.

Reruns are cheap: if a target directory already exists and has files in it,
the download is skipped unless --force is passed.

--- why group-e/traffic-lights-lxcaj is pinned to version 1, not 4 ---
Both version 4 (5096 images) and version 1 (12115 images) were downloaded
and their real data.yaml files inspected side by side before picking either
one blind (see PLAN.md's warning about "latest" exports being stripped,
which has already bitten this project once on indian_roads v7). Both
versions carry the SAME full 13-class list (Green, GreenLeft, GreenRight,
GreenStraight, GreenStraightLeft, GreenStraightRight, Red, RedLeft,
RedRight, RedStraight, RedStraightLeft, Yellow, off) -- so version 1 is not
a differently-scoped export, it is a strict superset of version 4 with
roughly 2.3x the annotations of every single class (Green 12105 vs 5200,
Red 7105 vs 3075, RedLeft 2540 vs 1092, Yellow 1023 vs 445, off 1633 vs
724, and so on down the small classes too). Since the whole point of this
dataset is maximising signal_red/signal_green training examples, version 1
strictly dominates version 4 here and there is no reason to take the
smaller one. Always check a newer version's data.yaml before bumping this
default -- "latest" has bitten this project before.
"""

import argparse
import os
import shutil

import yaml
from roboflow import Roboflow

WORKSPACE_PROJECT = {
    "signals_group_e": ("group-e", "traffic-lights-lxcaj", "data/datasets/signals_group_e"),
    "signals_detection": (
        "traffic-light-detection-975ge",
        "traffic-light-detection-l869b",
        "data/datasets/signals_detection",
    ),
    "signals_small": ("traffic-si0cm", "traffic-light-gxodz", "data/datasets/signals_small"),
}


def dir_is_nonempty(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def download_one(rf, workspace, project_name, version_num, target_dir, force):
    if dir_is_nonempty(target_dir) and not force:
        print(f"skipping {project_name}: {target_dir} already has files (use --force to redo)")
        return target_dir

    if force and os.path.isdir(target_dir):
        print(f"--force: removing existing {target_dir}")
        shutil.rmtree(target_dir)

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    # roboflow's download() insists on making its own leaf directory, so we
    # point it at the target directly (it creates target_dir itself).
    print(f"downloading {workspace}/{project_name} v{version_num} -> {target_dir}")
    version.download("yolov8", location=target_dir)

    return target_dir


def print_classes(target_dir, project_name):
    yaml_path = os.path.join(target_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        print(f"warning: no data.yaml found in {target_dir}")
        return

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    print(f"\n{project_name}: landed in {target_dir}")
    print(f"  {len(names)} classes found in data.yaml:")
    print(f"  {names}")


def main():
    parser = argparse.ArgumentParser(description="Download the Phase 3 signal-colour Roboflow datasets")
    parser.add_argument(
        "--group-e-version",
        type=int,
        default=1,
        help="group-e/traffic-lights-lxcaj version (default 1: the fuller 12115-image export, see docstring)",
    )
    parser.add_argument(
        "--detection-version",
        type=int,
        default=4,
        help="traffic-light-detection-975ge/traffic-light-detection-l869b version (default 4, latest)",
    )
    parser.add_argument(
        "--small-version",
        type=int,
        default=1,
        help="traffic-si0cm/traffic-light-gxodz version (default 1, only version published)",
    )
    parser.add_argument("--force", action="store_true", help="redownload even if the target dir already has files")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("error: ROBOFLOW_API_KEY is not set in the environment.")
        print("run like: ROBOFLOW_API_KEY=<key> .venv/bin/python scripts/download_signal_datasets.py")
        raise SystemExit(1)

    rf = Roboflow(api_key=api_key)

    jobs = [
        (*WORKSPACE_PROJECT["signals_group_e"], args.group_e_version),
        (*WORKSPACE_PROJECT["signals_detection"], args.detection_version),
        (*WORKSPACE_PROJECT["signals_small"], args.small_version),
    ]

    landed = []
    for workspace, project_name, target_dir, version_num in jobs:
        result_dir = download_one(rf, workspace, project_name, version_num, target_dir, args.force)
        landed.append((project_name, result_dir))

    for project_name, target_dir in landed:
        print_classes(target_dir, project_name)


if __name__ == "__main__":
    main()
