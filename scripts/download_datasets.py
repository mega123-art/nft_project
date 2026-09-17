"""
Phase 1 step 1-2: download the two public Roboflow datasets we need.

- indian-road-dataset/indian-roads-detection  (Indian roads, many classes)
- project-5wrkt/zebra-crossing-inwkp          (1 class, zebra crossings)

Both are pulled in YOLOv8 export format. The Roboflow API key is never
written to disk: it must be set in the environment as ROBOFLOW_API_KEY.

Reruns are cheap: if a target directory already exists and has files in it,
the download is skipped unless --force is passed.

--- why the Indian dataset version is pinned to 2, not "latest" ---
The Roboflow SDK's project.version() with no argument (or version 7, which
was "latest" when this was first written) gives a STRIPPED export with only
6 classes (Ambulance, Bus, Car, Tempo, Tractor, Truck) — no person, no
motorcycle/bike/cycle, no autorickshaw, no crosswalk, no traffic signal.
Version 2 of the same project has the full 48-class annotation set
(including Autorickshaw, MotorBike, person/Person, Zebra Crossing, Traffic
Signal, and the messy near-duplicate names PLAN.md warned about). If you
rerun this script and it silently grabs v7 again, half our unified classes
(motorcycle, autorickshaw, person, crosswalk) will get zero data again.
Always pass/keep --indian-version 2 unless you have specifically checked a
newer version's data.yaml and confirmed it still has the full class set.
"""

import argparse
import os
import shutil

import yaml
from roboflow import Roboflow

WORKSPACE_PROJECT = {
    "indian_roads": ("indian-road-dataset", "indian-roads-detection", "data/datasets/indian_roads"),
    "zebra_crossing": ("project-5wrkt", "zebra-crossing-inwkp", "data/datasets/zebra_crossing"),
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
    parser = argparse.ArgumentParser(description="Download the Phase 1 public Roboflow datasets")
    parser.add_argument("--indian-version", type=int, default=2, help="indian-roads-detection version (default 2: the full 48-class export, NOT latest)")
    parser.add_argument("--zebra-version", type=int, default=9, help="zebra-crossing-inwkp version (default 9)")
    parser.add_argument("--force", action="store_true", help="redownload even if the target dir already has files")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("error: ROBOFLOW_API_KEY is not set in the environment.")
        print("run like: ROBOFLOW_API_KEY=<key> .venv/bin/python scripts/download_datasets.py")
        raise SystemExit(1)

    rf = Roboflow(api_key=api_key)

    indian_ws, indian_proj, indian_dir = WORKSPACE_PROJECT["indian_roads"]
    zebra_ws, zebra_proj, zebra_dir = WORKSPACE_PROJECT["zebra_crossing"]

    landed = []
    target_dir = download_one(rf, indian_ws, indian_proj, args.indian_version, indian_dir, args.force)
    landed.append((indian_proj, target_dir))

    target_dir = download_one(rf, zebra_ws, zebra_proj, args.zebra_version, zebra_dir, args.force)
    landed.append((zebra_proj, target_dir))

    for project_name, target_dir in landed:
        print_classes(target_dir, project_name)


if __name__ == "__main__":
    main()
