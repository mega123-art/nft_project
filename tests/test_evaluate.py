"""
Phase 8 tests for scripts/evaluate.py's metric computations.

Synthetic throughout: small hand-built sqlite DBs (via decision_log.py's own
DecisionLogger, so the schema tested here is the schema evaluate.py will
actually see in practice) and small hand-built ground-truth CSVs. The
"ground truth absent" path is tested explicitly and must produce the
NOT COMPUTED marker rather than a number -- that is the main risk this
script is built to avoid (see its own module docstring).
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fsm import Verdict, SEARCHING, WAITING, SAFE_TO_CROSS
from detector import Detection
from decision_log import DecisionLogger

import evaluate


def make_verdict(state, min_ttc=None):
    reason = {
        SEARCHING: "looking for a crossing",
        WAITING: "wait, vehicle approaching",
        SAFE_TO_CROSS: "safe to cross now",
    }[state]
    return Verdict(state=state, reason=reason, raw_state=state, raw_reason=reason,
                    min_ttc=min_ttc, crosswalk_offset=None, crosswalk_direction=None, changed=True)


def build_db(tmp_path, rows):
    """rows: list of (frame_index, state, min_ttc, video_source). Writes a
    real decisions.db via DecisionLogger so the schema matches production."""
    db_path = str(tmp_path / "decisions.db")
    logger = DecisionLogger(db_path=db_path, video_source=None, batch_size=1)
    for frame_index, state, min_ttc, video_source in rows:
        logger.video_source = video_source
        logger.log(frame_index, make_verdict(state, min_ttc=min_ttc),
                   unresolved_vehicles=0, detections=[Detection("car", 0.9, 0, 0, 10, 10, 1)],
                   inference_ms=20.0, total_ms=40.0)
    logger.close()
    return db_path


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# False-safe rate
# ---------------------------------------------------------------------------

def test_false_safe_rate_not_computed_without_ground_truth(tmp_path):
    db_path = build_db(tmp_path, [(1, SAFE_TO_CROSS, None, "clip.mp4")])
    decisions = evaluate.load_decisions(db_path)
    frame_labels = evaluate.load_frame_labels(str(tmp_path / "missing.csv"))

    result = evaluate.compute_false_safe_rate(decisions, frame_labels)
    assert result["computed"] is False
    assert "no ground truth" in result["reason"]


def test_false_safe_rate_zero_when_all_safe_frames_are_actually_safe(tmp_path):
    db_path = build_db(tmp_path, [
        (1, SAFE_TO_CROSS, None, "clip.mp4"),
        (2, WAITING, 3.0, "clip.mp4"),
    ])
    labels_path = tmp_path / "frame_labels.csv"
    write_csv(labels_path, ["video_source", "start_frame", "end_frame", "label"], [
        {"video_source": "clip.mp4", "start_frame": 1, "end_frame": 1, "label": "safe"},
        {"video_source": "clip.mp4", "start_frame": 2, "end_frame": 2, "label": "unsafe"},
    ])

    decisions = evaluate.load_decisions(db_path)
    frame_labels = evaluate.load_frame_labels(str(labels_path))
    result = evaluate.compute_false_safe_rate(decisions, frame_labels)

    assert result["computed"] is True
    assert result["false_safe_frames"] == 0
    assert result["labelled_frames"] == 2
    assert result["rate"] == 0.0


def test_false_safe_rate_detects_a_real_false_safe(tmp_path):
    db_path = build_db(tmp_path, [
        (1, SAFE_TO_CROSS, None, "clip.mp4"),  # system says safe
    ])
    labels_path = tmp_path / "frame_labels.csv"
    write_csv(labels_path, ["video_source", "start_frame", "end_frame", "label"], [
        {"video_source": "clip.mp4", "start_frame": 1, "end_frame": 1, "label": "unsafe"},  # but it wasn't
    ])

    decisions = evaluate.load_decisions(db_path)
    frame_labels = evaluate.load_frame_labels(str(labels_path))
    result = evaluate.compute_false_safe_rate(decisions, frame_labels)

    assert result["computed"] is True
    assert result["false_safe_frames"] == 1
    assert result["labelled_frames"] == 1
    assert result["rate"] == 1.0
    assert result["examples"] == [("clip.mp4", 1)]


def test_false_safe_rate_not_computed_when_video_source_names_dont_match(tmp_path):
    db_path = build_db(tmp_path, [(1, SAFE_TO_CROSS, None, "clip.mp4")])
    labels_path = tmp_path / "frame_labels.csv"
    write_csv(labels_path, ["video_source", "start_frame", "end_frame", "label"], [
        {"video_source": "other_clip.mp4", "start_frame": 1, "end_frame": 1, "label": "unsafe"},
    ])

    decisions = evaluate.load_decisions(db_path)
    frame_labels = evaluate.load_frame_labels(str(labels_path))
    result = evaluate.compute_false_safe_rate(decisions, frame_labels)
    assert result["computed"] is False


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def test_latency_not_computed_on_empty_db(tmp_path):
    result = evaluate.compute_latency_stats([])
    assert result["computed"] is False


def test_latency_percentiles(tmp_path):
    rows = [{"total_ms": v} for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    result = evaluate.compute_latency_stats(rows)
    assert result["computed"] is True
    assert result["mean"] == pytest.approx(55.0)
    assert result["max"] == 100
    assert result["p50"] == pytest.approx(55.0)
    assert result["meets_target_p95"] is True  # all well under 300ms


def test_latency_misses_target_when_p95_over_300ms():
    # Every value is well past the 300ms target, so p95 unambiguously is too
    # -- avoids relying on exact interpolation behaviour at a small n.
    rows = [{"total_ms": 500.0} for _ in range(20)]
    result = evaluate.compute_latency_stats(rows)
    assert result["computed"] is True
    assert result["p95"] > 300.0
    assert result["meets_target_p95"] is False


# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------

def test_frame_rate_not_computed_with_no_rows():
    result = evaluate.compute_frame_rate([])
    assert result["computed"] is False


def test_frame_rate_computed_from_wall_time_span():
    # 11 frames spanning exactly 10 seconds => 1.0 fps.
    rows = [{"video_source": "clip.mp4", "wall_time": float(i)} for i in range(11)]
    result = evaluate.compute_frame_rate(rows)
    assert result["computed"] is True
    assert result["per_video"]["clip.mp4"]["frames"] == 11
    assert result["per_video"]["clip.mp4"]["fps"] == pytest.approx(1.0)
    assert result["overall_fps"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TTC error
# ---------------------------------------------------------------------------

def test_ttc_error_not_computed_without_ground_truth():
    result = evaluate.compute_ttc_error([], None)
    assert result["computed"] is False


def test_ttc_error_computes_signed_and_abs_error(tmp_path):
    db_path = build_db(tmp_path, [(10, WAITING, 4.5, "clip.mp4")])
    decisions = evaluate.load_decisions(db_path)

    obs_path = tmp_path / "ttc_observations.csv"
    write_csv(obs_path, ["video_source", "frame_index", "true_ttc_s"], [
        {"video_source": "clip.mp4", "frame_index": 10, "true_ttc_s": 4.0},
    ])
    ttc_observations = evaluate.load_ttc_observations(str(obs_path))

    result = evaluate.compute_ttc_error(decisions, ttc_observations)
    assert result["computed"] is True
    assert result["n_matched"] == 1
    assert result["matched"][0]["error"] == pytest.approx(0.5)
    assert result["mean_abs_error_s"] == pytest.approx(0.5)


def test_ttc_error_separates_no_ttc_and_unmatched(tmp_path):
    db_path = build_db(tmp_path, [(10, SEARCHING, None, "clip.mp4")])  # min_ttc NULL
    decisions = evaluate.load_decisions(db_path)

    obs_path = tmp_path / "ttc_observations.csv"
    write_csv(obs_path, ["video_source", "frame_index", "true_ttc_s"], [
        {"video_source": "clip.mp4", "frame_index": 10, "true_ttc_s": 4.0},  # logged, but null ttc
        {"video_source": "clip.mp4", "frame_index": 999, "true_ttc_s": 2.0},  # never logged
    ])
    ttc_observations = evaluate.load_ttc_observations(str(obs_path))

    result = evaluate.compute_ttc_error(decisions, ttc_observations)
    assert result["computed"] is True
    assert result["n_matched"] == 0
    assert result["n_reported_no_ttc"] == 1
    assert result["n_unmatched_frame"] == 1
    assert "mean_abs_error_s" not in result


# ---------------------------------------------------------------------------
# mAP50 table
# ---------------------------------------------------------------------------

def test_map_table_reads_existing_metrics_json(tmp_path):
    a = tmp_path / "a.json"
    a.write_text('{"per_class_map50": {"car": 0.9, "bus": null}}')
    missing_path = str(tmp_path / "missing.json")
    map_sources = evaluate.load_map_metrics([("Run A", str(a)), ("Run B", missing_path)])
    result = evaluate.compute_map_table(map_sources)
    assert result["computed"] is True
    assert result["table"]["car"]["Run A"] == 0.9
    assert result["table"]["bus"]["Run A"] is None
    assert ("Run B", missing_path) in result["missing_runs"]


def test_map_table_not_computed_when_nothing_found(tmp_path):
    map_sources = evaluate.load_map_metrics([("Run A", str(tmp_path / "missing.json"))])
    result = evaluate.compute_map_table(map_sources)
    assert result["computed"] is False


# ---------------------------------------------------------------------------
# Loaders: missing vs empty-but-present ground truth
# ---------------------------------------------------------------------------

def test_load_frame_labels_returns_none_when_file_missing(tmp_path):
    assert evaluate.load_frame_labels(str(tmp_path / "nope.csv")) is None


def test_load_frame_labels_returns_dict_when_file_present_but_empty(tmp_path):
    path = tmp_path / "frame_labels.csv"
    write_csv(path, ["video_source", "start_frame", "end_frame", "label"], [])
    result = evaluate.load_frame_labels(str(path))
    assert result == {}


def test_load_ttc_observations_returns_none_when_file_missing(tmp_path):
    assert evaluate.load_ttc_observations(str(tmp_path / "nope.csv")) is None
