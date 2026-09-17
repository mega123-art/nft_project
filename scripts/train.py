"""
Phase 2 step 1: thin wrapper around Ultralytics training.

Deliberately does almost nothing beyond calling YOLO().train() with CLI
args -- the whole point is that this script runs identically on Colab
(free T4 GPU) and locally on CPU, and someone reading it in a viva should
be able to explain every line.

After training, Ultralytics writes its own results (weights, PR curves,
confusion matrix, etc.) under runs/detect/<name>/. This script additionally
pulls the per-class mAP50 out of the validation results and writes it to
report/phase2_metrics.json, so the numbers for the report come straight
from the run instead of being copied out of terminal scrollback by hand.
"""

import argparse
import json
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train the road-crossing detector (Phase 2 baseline)")
    parser.add_argument("--data", default="data/data.yaml", help="path to the dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="base weights to start from")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="phase2_baseline", help="run name under runs/detect/")
    parser.add_argument(
        "--metrics-out",
        default="report/phase2_metrics.json",
        help="where to write the per-class mAP50 summary",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    # model.train() already runs a final validation pass and returns its
    # metrics, so we don't need a separate model.val() call here.
    class_names = results.names  # {0: "car", 1: "bus", ...}
    per_class_map50 = {}
    for i, class_id in enumerate(results.ap_class_index):
        name = class_names[int(class_id)]
        per_class_map50[name] = float(results.box.ap50[i])

    # classes with zero validation instances (bus, signal_red, signal_green
    # in Phase 2 -- see PLAN.md's known data gaps) never appear in
    # ap_class_index at all, so fill them in explicitly rather than silently
    # leaving them out of the report.
    for name in class_names.values():
        if name not in per_class_map50:
            per_class_map50[name] = None

    summary = {
        "run_name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "per_class_map50": per_class_map50,
    }

    print("\nper-class mAP50:")
    for name, value in per_class_map50.items():
        value_str = f"{value:.4f}" if value is not None else "no val instances"
        print(f"  {name:<14} {value_str}")
    print(f"\noverall mAP50:    {summary['map50']:.4f}")
    print(f"overall mAP50-95: {summary['map50_95']:.4f}")

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote metrics summary to {args.metrics_out}")


if __name__ == "__main__":
    main()
