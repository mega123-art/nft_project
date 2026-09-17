"""
Phase 2 helper: zip up the merged dataset for uploading to Google Drive,
so notebooks/phase2_train_colab.ipynb has something to unzip on Colab.

We do this instead of re-downloading from Roboflow inside the notebook
because that would mean putting the Roboflow API key into a notebook file.

Usage:
    .venv/bin/python scripts/make_dataset_zip.py
    (writes data/datasets/merged_dataset.zip, ~242MB)
"""

import argparse
import os
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Zip the merged dataset for Colab upload")
    parser.add_argument("--source", default="data/datasets/merged", help="merged dataset directory")
    parser.add_argument("--out", default="data/datasets/merged_dataset.zip", help="output zip path")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"error: {args.source} does not exist -- run merge_datasets.py first")
        raise SystemExit(1)

    if os.path.exists(args.out):
        os.remove(args.out)

    file_count = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(args.source):
            for name in files:
                full_path = os.path.join(root, name)
                # store paths relative to the source dir, e.g. "train/images/x.jpg",
                # so unzipping on Colab produces train/ and val/ directly.
                arcname = os.path.relpath(full_path, args.source)
                zf.write(full_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"wrote {args.out} ({file_count} files, {size_mb:.1f} MB)")
    print("upload this zip to Google Drive, then point the Colab notebook at it")


if __name__ == "__main__":
    main()
