"""
Phase 8: turn logs/decisions.db (+ optional hand-annotated ground truth)
into the metrics PLAN.md's Phase 8 section asks for, and a markdown summary
at report/evaluation.md.

WHAT THIS SCRIPT WILL AND WILL NOT DO. Two of PLAN.md's five metrics --
false-safe rate and TTC error -- need ground truth that nobody has produced
yet (data/ground_truth/*.csv; see data/ground_truth/SCHEMA.md for the
format, and scripts/annotate_ground_truth.py for a way to produce the
frame-label file without hand-counting frame numbers in a video player).
This script does NOT invent a ground truth, and does NOT quietly report
0.0 for either metric when the ground-truth file is simply missing --
a false-safe rate of 0.0 computed against zero labelled frames looks
identical to a false-safe rate of 0.0 computed against a thousand
carefully checked frames, and only one of those numbers means anything.
When a ground-truth file is missing, the corresponding section of both the
terminal output and report/evaluation.md says plainly:

    NOT COMPUTED -- no ground truth at <path>. <what to do about it>

rather than a number. Per-class mAP50 and (frame rate + latency) ARE always
computed here, because their inputs already exist: the report/*_metrics.json
files from earlier training runs, and logs/decisions.db from any main.py run
with logging enabled.

WHY frame_index, NOT wall-clock time, IS THE JOIN KEY for ground truth.
decisions.db logs one row per processed frame, keyed by (video_source,
frame_index) -- the same numbering main.py's own frame_count uses. A human
annotator scrubbing the same video file in any ordinary player sees that
same frame count (or a very close, deterministic function of it via fps),
which is something a wall-clock timestamp comparison could never guarantee
across two different runs of the pipeline at two different frame rates.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fsm import SAFE_TO_CROSS  # noqa: E402  (path insert must happen first)

DEFAULT_DB_PATH = str(REPO_ROOT / "logs" / "decisions.db")
DEFAULT_FRAME_LABELS = str(REPO_ROOT / "data" / "ground_truth" / "frame_labels.csv")
DEFAULT_TTC_OBSERVATIONS = str(REPO_ROOT / "data" / "ground_truth" / "ttc_observations.csv")
DEFAULT_OUT = str(REPO_ROOT / "report" / "evaluation.md")

# (display name, path) -- the three training runs PLAN.md's Phase 8 wants
# compared. All three already exist on disk from earlier phases; nothing
# here regenerates them.
DEFAULT_MAP_SOURCES = [
    ("Phase 2 (public data only)", str(REPO_ROOT / "report" / "phase2_metrics.json")),
    ("Phase 3 sub (+signals, capped 1.5k)", str(REPO_ROOT / "report" / "phase3_sub_metrics.json")),
    ("Phase 3 full (+signals, all 8.1k)", str(REPO_ROOT / "report" / "phase3_full_metrics.json")),
]

NOT_COMPUTED = "NOT_COMPUTED"  # sentinel; never a real metric value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_decisions(db_path):
    """Return all rows from decisions.db as a list of dicts, oldest first.
    [] if the file doesn't exist -- callers decide what that means for each
    metric rather than this function guessing."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM decisions ORDER BY id ASC").fetchall()
    except sqlite3.OperationalError:
        # File exists but has no decisions table yet (e.g. touched by hand).
        rows = []
    conn.close()
    return [dict(r) for r in rows]


def load_frame_labels(path):
    """Return {video_source: [(start_frame, end_frame, label), ...]}, or
    None if the file doesn't exist -- None is the "NOT COMPUTED" signal,
    distinct from an empty-but-present file (which is a real, if useless,
    ground truth of zero labelled frames)."""
    if not os.path.exists(path):
        return None
    by_video = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_video[row["video_source"]].append(
                (int(row["start_frame"]), int(row["end_frame"]), row["label"].strip().lower())
            )
    return dict(by_video)


def load_ttc_observations(path):
    """Return a list of (video_source, frame_index, true_ttc_s) dicts, or
    None if the file doesn't exist. See load_frame_labels for why None (not
    []) is the "missing" signal."""
    if not os.path.exists(path):
        return None
    observations = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            observations.append({
                "video_source": row["video_source"],
                "frame_index": int(row["frame_index"]),
                "true_ttc_s": float(row["true_ttc_s"]),
            })
    return observations


def load_map_metrics(sources):
    """Read each (name, path) pair's JSON if present. Returns a list of
    (name, path, data_or_None) -- missing files are reported, not silently
    skipped, since a report that just drops a column looks like the run
    never existed rather than like the file wasn't found."""
    results = []
    for name, path in sources:
        if os.path.exists(path):
            with open(path) as f:
                results.append((name, path, json.load(f)))
        else:
            results.append((name, path, None))
    return results


# ---------------------------------------------------------------------------
# Metric: false-safe rate
# ---------------------------------------------------------------------------

def _label_for_frame(ranges, frame_index):
    """ranges is a list of (start, end, label) for one video. Returns the
    label covering frame_index, or None if uncovered. Assumes
    non-overlapping ranges (see SCHEMA.md); the first match wins if that
    assumption is violated, which is a labelling bug to fix, not something
    this function tries to arbitrate."""
    for start, end, label in ranges:
        if start <= frame_index <= end:
            return label
    return None


def compute_false_safe_rate(decisions, frame_labels):
    """Returns a dict describing the false-safe rate, or {"computed": False,
    "reason": ...} when frame_labels is None (no ground truth at all).

    Definition used here (documented because PLAN.md states the goal, not
    the exact denominator): among all frames that HAVE a ground-truth
    label, the fraction where the system's committed state was
    SAFE_TO_CROSS while the label says "unsafe". Frames with no ground
    truth are excluded from both numerator and denominator -- they are not
    evidence of anything, the same principle safety.py applies to its own
    UNKNOWN status.
    """
    if frame_labels is None:
        return {
            "computed": False,
            "reason": f"no ground truth at {DEFAULT_FRAME_LABELS} (see data/ground_truth/SCHEMA.md)",
        }

    labelled_frames = 0
    false_safe_frames = 0
    safe_and_labelled_unsafe = []  # for the report: which frames, for spot-checking

    for row in decisions:
        ranges = frame_labels.get(row["video_source"])
        if not ranges:
            continue
        label = _label_for_frame(ranges, row["frame_index"])
        if label is None:
            continue
        labelled_frames += 1
        if label == "unsafe" and row["state"] == SAFE_TO_CROSS:
            false_safe_frames += 1
            safe_and_labelled_unsafe.append((row["video_source"], row["frame_index"]))

    if labelled_frames == 0:
        return {
            "computed": False,
            "reason": (
                "ground-truth file exists but no logged frame in decisions.db "
                "falls inside any labelled range (check video_source spelling "
                "matches between the CSV and the DB)"
            ),
        }

    return {
        "computed": True,
        "false_safe_frames": false_safe_frames,
        "labelled_frames": labelled_frames,
        "rate": false_safe_frames / labelled_frames,
        "examples": safe_and_labelled_unsafe[:10],
    }


# ---------------------------------------------------------------------------
# Metric: end-to-end latency (capture -> audio start)
# ---------------------------------------------------------------------------

def _percentile(sorted_values, pct):
    """Linear-interpolation percentile, pct in [0, 100]. sorted_values must
    be non-empty and already sorted."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def compute_latency_stats(decisions):
    """Distribution of total_ms (frame capture through audio-start, see
    main.py's comment at the call site) across all logged frames. Reports
    p50/p95/max, not just the mean -- PLAN.md's 300ms target is explicitly
    about the tail, and a mean can hide a bad tail entirely."""
    values = sorted(r["total_ms"] for r in decisions if r["total_ms"] is not None)
    if not values:
        return {"computed": False, "reason": "no logged frames with a total_ms value"}
    return {
        "computed": True,
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": values[-1],
        "target_ms": 300.0,
        "meets_target_p95": _percentile(values, 95) < 300.0,
    }


# ---------------------------------------------------------------------------
# Metric: frame rate
# ---------------------------------------------------------------------------

def compute_frame_rate(decisions):
    """fps per video_source, from the real wall-clock span the logger
    stamped on each row (wall_time), not from an assumed constant frame
    interval -- this pipeline's fps is variable (see safety.py's own
    variable-dt handling), so only elapsed real time gives an honest
    number."""
    if not decisions:
        return {"computed": False, "reason": "no logged frames"}

    by_video = defaultdict(list)
    for row in decisions:
        by_video[row["video_source"]].append(row["wall_time"])

    per_video = {}
    for video, times in by_video.items():
        times.sort()
        n = len(times)
        span = times[-1] - times[0]
        per_video[video] = {
            "frames": n,
            "fps": (n - 1) / span if span > 0 else None,
        }

    all_times = sorted(r["wall_time"] for r in decisions)
    overall_span = all_times[-1] - all_times[0]
    overall_fps = (len(all_times) - 1) / overall_span if overall_span > 0 else None

    return {"computed": True, "per_video": per_video, "overall_fps": overall_fps}


# ---------------------------------------------------------------------------
# Metric: TTC error against manual ground truth
# ---------------------------------------------------------------------------

def compute_ttc_error(decisions, ttc_observations):
    if ttc_observations is None:
        return {
            "computed": False,
            "reason": f"no ground truth at {DEFAULT_TTC_OBSERVATIONS} (see data/ground_truth/SCHEMA.md)",
        }

    by_key = {(r["video_source"], r["frame_index"]): r for r in decisions}

    matched = []
    unmatched_frame = []      # observation's frame never logged at all
    reported_no_ttc = []      # logged, but min_ttc was NULL that frame

    for obs in ttc_observations:
        key = (obs["video_source"], obs["frame_index"])
        row = by_key.get(key)
        if row is None:
            unmatched_frame.append(obs)
            continue
        if row["min_ttc"] is None:
            reported_no_ttc.append(obs)
            continue
        error = row["min_ttc"] - obs["true_ttc_s"]
        matched.append({
            "video_source": obs["video_source"],
            "frame_index": obs["frame_index"],
            "predicted_ttc": row["min_ttc"],
            "true_ttc": obs["true_ttc_s"],
            "error": error,
        })

    if not matched and not reported_no_ttc:
        return {
            "computed": False,
            "reason": (
                "ground-truth file exists but none of its (video_source, "
                "frame_index) pairs match a logged frame in decisions.db"
            ),
        }

    abs_errors = sorted(abs(m["error"]) for m in matched)
    result = {
        "computed": True,
        "n_observations": len(ttc_observations),
        "n_matched": len(matched),
        "n_unmatched_frame": len(unmatched_frame),
        "n_reported_no_ttc": len(reported_no_ttc),
        "matched": matched,
    }
    if abs_errors:
        result["mean_abs_error_s"] = sum(abs_errors) / len(abs_errors)
        result["median_abs_error_s"] = _percentile(abs_errors, 50)
        result["max_abs_error_s"] = abs_errors[-1]
    return result


# ---------------------------------------------------------------------------
# Metric: per-class mAP50 comparison table
# ---------------------------------------------------------------------------

def compute_map_table(map_sources):
    """Build a class -> {run_name: map50} table across all runs that were
    actually found on disk. Missing runs are recorded, not silently
    dropped, so the report says WHICH comparison points are absent."""
    runs = []
    missing = []
    class_names = []
    for name, path, data in map_sources:
        if data is None:
            missing.append((name, path))
            continue
        runs.append((name, data))
        for cls in data.get("per_class_map50", {}):
            if cls not in class_names:
                class_names.append(cls)

    table = {cls: {} for cls in class_names}
    for name, data in runs:
        per_class = data.get("per_class_map50", {})
        for cls in class_names:
            table[cls][name] = per_class.get(cls)  # may be None (class absent that run)

    return {
        "computed": bool(runs),
        "class_names": class_names,
        "run_names": [name for name, _ in runs],
        "table": table,
        "missing_runs": missing,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(value, spec="{:.3f}"):
    return spec.format(value) if value is not None else "--"


def render_report(false_safe, latency, frame_rate, ttc_error, map_table, db_path):
    lines = []
    lines.append("# Phase 8 evaluation report")
    lines.append("")
    lines.append(f"Generated from `{db_path}` by `scripts/evaluate.py`.")
    lines.append("")

    # -- False-safe rate: first and most prominent, per PLAN.md. -----------
    lines.append("## False-safe rate (most important number in this report)")
    lines.append("")
    if false_safe["computed"]:
        lines.append(
            f"**{false_safe['rate']:.4f}** "
            f"({false_safe['false_safe_frames']} / {false_safe['labelled_frames']} labelled frames) "
            "-- target is 0."
        )
        if false_safe["false_safe_frames"]:
            lines.append("")
            lines.append("Example false-safe frames (video, frame_index):")
            for video, frame_idx in false_safe["examples"]:
                lines.append(f"- {video}, frame {frame_idx}")
    else:
        lines.append(f"**NOT COMPUTED** -- {false_safe['reason']}.")
        lines.append("")
        lines.append(
            "To unblock: annotate `data/ground_truth/frame_labels.csv` "
            "(schema in `data/ground_truth/SCHEMA.md`), e.g. via "
            "`python scripts/annotate_ground_truth.py <video> --video-source <name>` "
            "while `--log-db` was pointed at the same DB this report reads."
        )
    lines.append("")

    # -- Latency -------------------------------------------------------
    lines.append("## End-to-end latency (frame capture -> audio start)")
    lines.append("")
    if latency["computed"]:
        lines.append(f"n = {latency['n']} logged frames")
        lines.append("")
        lines.append("| stat | ms |")
        lines.append("|---|---|")
        lines.append(f"| mean | {latency['mean']:.1f} |")
        lines.append(f"| p50 | {latency['p50']:.1f} |")
        lines.append(f"| p95 | {latency['p95']:.1f} |")
        lines.append(f"| max | {latency['max']:.1f} |")
        verdict = "MEETS" if latency["meets_target_p95"] else "MISSES"
        lines.append("")
        lines.append(f"Target: p95 < {latency['target_ms']:.0f} ms -- **{verdict}** target.")
    else:
        lines.append(f"**NOT COMPUTED** -- {latency['reason']}.")
    lines.append("")

    # -- Frame rate ------------------------------------------------------
    lines.append("## Frame rate")
    lines.append("")
    if frame_rate["computed"]:
        lines.append("| video_source | frames | fps |")
        lines.append("|---|---|---|")
        for video, stats in frame_rate["per_video"].items():
            lines.append(f"| {video} | {stats['frames']} | {_fmt(stats['fps'], '{:.2f}')} |")
        lines.append("")
        lines.append(f"Overall: {_fmt(frame_rate['overall_fps'], '{:.2f}')} fps "
                      f"across all logged frames.")
    else:
        lines.append(f"**NOT COMPUTED** -- {frame_rate['reason']}.")
    lines.append("")

    # -- TTC error ---------------------------------------------------------
    lines.append("## TTC error vs. manually annotated ground truth")
    lines.append("")
    if ttc_error["computed"]:
        lines.append(
            f"{ttc_error['n_matched']} / {ttc_error['n_observations']} annotated observations "
            f"matched a logged frame with a non-null min_ttc "
            f"({ttc_error['n_unmatched_frame']} had no matching logged frame at all, "
            f"{ttc_error['n_reported_no_ttc']} matched a frame where the system reported no TTC)."
        )
        lines.append("")
        if "mean_abs_error_s" in ttc_error:
            lines.append("| stat | seconds |")
            lines.append("|---|---|")
            lines.append(f"| mean abs error | {ttc_error['mean_abs_error_s']:.2f} |")
            lines.append(f"| median abs error | {ttc_error['median_abs_error_s']:.2f} |")
            lines.append(f"| max abs error | {ttc_error['max_abs_error_s']:.2f} |")
            lines.append("")
            lines.append("| video | frame | predicted TTC (s) | true TTC (s) | error (s) |")
            lines.append("|---|---|---|---|---|")
            for m in ttc_error["matched"]:
                lines.append(
                    f"| {m['video_source']} | {m['frame_index']} | "
                    f"{m['predicted_ttc']:.2f} | {m['true_ttc']:.2f} | {m['error']:+.2f} |"
                )
        else:
            lines.append(
                "All annotated observations landed on frames where the system reported no "
                "TTC (min_ttc was NULL) -- error cannot be computed from these observations."
            )
    else:
        lines.append(f"**NOT COMPUTED** -- {ttc_error['reason']}.")
        lines.append("")
        lines.append(
            "To unblock: annotate `data/ground_truth/ttc_observations.csv` "
            "(schema in `data/ground_truth/SCHEMA.md`) against ~20 vehicle "
            "tracks, matched by (video_source, frame_index) to rows already "
            "in decisions.db."
        )
    lines.append("")

    # -- Per-class mAP50 comparison -----------------------------------------
    lines.append("## Per-class mAP50 comparison (Phase 2 vs Phase 3 sub vs Phase 3 full)")
    lines.append("")
    lines.append(
        "This is PLAN.md's \"strongest graph in the report\" -- it is a table "
        "here; plot it directly from these numbers for the write-up."
    )
    lines.append("")
    if map_table["computed"]:
        header = "| class | " + " | ".join(map_table["run_names"]) + " |"
        sep = "|---|" + "|".join(["---"] * len(map_table["run_names"])) + "|"
        lines.append(header)
        lines.append(sep)
        for cls in map_table["class_names"]:
            row_vals = [_fmt(map_table["table"][cls].get(name)) for name in map_table["run_names"]]
            lines.append(f"| {cls} | " + " | ".join(row_vals) + " |")
    else:
        lines.append("**NOT COMPUTED** -- none of the expected metrics JSON files were found.")
    if map_table["missing_runs"]:
        lines.append("")
        lines.append("Missing runs (expected but not found on disk):")
        for name, path in map_table["missing_runs"]:
            lines.append(f"- {name}: `{path}`")
        lines.append(
            "\nNote: Phase 4 (\"our own footage\") has not been trained yet, so a "
            "fourth comparison column for it does not exist -- this table currently "
            "only spans Phase 2 and Phase 3, not the full Phase 2/3/4 progression "
            "PLAN.md's Phase 8 section describes."
        )
    lines.append("")

    # -- Things this script deliberately does not attempt -------------------
    lines.append("## Confusion matrix and PR curves")
    lines.append("")
    lines.append(
        "Not regenerated here. Ultralytics writes these automatically under "
        "`runs/detect/<run_name>/{confusion_matrix.png,PR_curve.png,...}` as "
        "a side effect of `scripts/train.py` (or a plain `model.val()` call) "
        "-- but the training pod that produced Phase 2/3's `runs/` directory "
        "has been destroyed, and that directory is gitignored, so those "
        "artifacts do not exist anywhere in this repo right now. They need a "
        "fresh training or validation run (on any machine with the dataset "
        "and `models/best.pt`) to regenerate; running "
        "`model.val(data=\"data/data.yaml\")` alone is enough, a full retrain "
        "is not required."
    )
    lines.append("")
    lines.append("## Annotated screenshots")
    lines.append("")
    lines.append(
        "Not generated by this script. `report/phase6_fsm_check/` and "
        "`report/phase2_best_check/` already contain a handful of annotated "
        "frames from earlier phases (state + reason drawn, per Phase 6's own "
        "done-check) -- reuse `main.py --save-frames DIR` against clips that "
        "hit each of SEARCHING / WAITING / SAFE_TO_CROSS to get the 6-8 "
        "screenshots PLAN.md asks for; none of the current sample clips are "
        "confirmed to reach SAFE_TO_CROSS (see the ground-truth note above "
        "on why -- sparse detections on non-Indian CCTV footage)."
    )
    lines.append("")

    return "\n".join(lines)


def print_summary(false_safe, latency, frame_rate, ttc_error, map_table):
    print("=" * 70)
    print("FALSE-SAFE RATE (most important number -- target 0):")
    if false_safe["computed"]:
        print(f"  {false_safe['rate']:.4f}  "
              f"({false_safe['false_safe_frames']} / {false_safe['labelled_frames']} labelled frames)")
    else:
        print(f"  NOT COMPUTED -- {false_safe['reason']}")
    print("=" * 70)

    print("\nEnd-to-end latency (capture -> audio start):")
    if latency["computed"]:
        print(f"  mean={latency['mean']:.1f}ms p50={latency['p50']:.1f}ms "
              f"p95={latency['p95']:.1f}ms max={latency['max']:.1f}ms "
              f"(n={latency['n']})")
        print(f"  target p95 < 300ms: {'MEETS' if latency['meets_target_p95'] else 'MISSES'}")
    else:
        print(f"  NOT COMPUTED -- {latency['reason']}")

    print("\nFrame rate:")
    if frame_rate["computed"]:
        for video, stats in frame_rate["per_video"].items():
            print(f"  {video}: {stats['frames']} frames, {_fmt(stats['fps'], '{:.2f}')} fps")
        print(f"  overall: {_fmt(frame_rate['overall_fps'], '{:.2f}')} fps")
    else:
        print(f"  NOT COMPUTED -- {frame_rate['reason']}")

    print("\nTTC error vs ground truth:")
    if ttc_error["computed"] and "mean_abs_error_s" in ttc_error:
        print(f"  mean abs error={ttc_error['mean_abs_error_s']:.2f}s "
              f"median={ttc_error['median_abs_error_s']:.2f}s "
              f"max={ttc_error['max_abs_error_s']:.2f}s "
              f"(matched {ttc_error['n_matched']}/{ttc_error['n_observations']})")
    elif ttc_error["computed"]:
        print("  all matched observations had a null system TTC -- no error computable")
    else:
        print(f"  NOT COMPUTED -- {ttc_error['reason']}")

    print("\nPer-class mAP50 comparison:")
    if map_table["computed"]:
        print(f"  runs: {', '.join(map_table['run_names'])}")
        for cls in map_table["class_names"]:
            vals = " ".join(f"{name}={_fmt(map_table['table'][cls].get(name))}" for name in map_table["run_names"])
            print(f"  {cls}: {vals}")
    else:
        print("  NOT COMPUTED -- no metrics JSON files found")
    if map_table["missing_runs"]:
        print(f"  (missing: {', '.join(name for name, _ in map_table['missing_runs'])})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Phase 8 evaluation report generator")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="decisions.db path")
    parser.add_argument("--frame-labels", default=DEFAULT_FRAME_LABELS,
                         help="ground-truth safe/unsafe frame ranges CSV")
    parser.add_argument("--ttc-observations", default=DEFAULT_TTC_OBSERVATIONS,
                         help="ground-truth TTC observations CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="markdown report output path")
    args = parser.parse_args()

    decisions = load_decisions(args.db)
    if not decisions:
        print(f"warning: no rows found in {args.db} -- did you run main.py with logging enabled?")

    frame_labels = load_frame_labels(args.frame_labels)
    ttc_observations = load_ttc_observations(args.ttc_observations)
    map_sources = load_map_metrics(DEFAULT_MAP_SOURCES)

    false_safe = compute_false_safe_rate(decisions, frame_labels)
    latency = compute_latency_stats(decisions)
    frame_rate = compute_frame_rate(decisions)
    ttc_error = compute_ttc_error(decisions, ttc_observations)
    map_table = compute_map_table(map_sources)

    print_summary(false_safe, latency, frame_rate, ttc_error, map_table)

    report_md = render_report(false_safe, latency, frame_rate, ttc_error, map_table, args.db)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report_md)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
