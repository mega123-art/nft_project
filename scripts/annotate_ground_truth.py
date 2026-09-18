"""
Phase 8 helper: step through a clip and mark unsafe frame ranges by hand,
writing data/ground_truth/frame_labels.csv (see that directory's SCHEMA.md
for the format scripts/evaluate.py actually reads).

This exists because "open the CSV in a text editor and count frame numbers
against a video player" is how ground-truth annotation quietly never
happens. This script is deliberately small: it does not try to be a real
video annotation tool, it just turns keypresses into CSV rows while you
watch the same clip main.py would process.

CONTROLS (window must have focus):
    space       play / pause
    n           step forward one frame (while paused)
    u           toggle "currently unsafe" on/off -- pressing it opens an
                unsafe range starting at the frame on screen; pressing it
                again closes the range at the frame on screen (inclusive)
    q           quit and write out whatever ranges were closed so far
                (an unsafe range left open at quit time is closed at the
                last frame seen, so you never lose a range by forgetting
                to press 'u' again before quitting)

Frames not covered by any recorded unsafe range are written as a single
trailing "safe" range covering everything else in the clip -- see
SCHEMA.md: evaluate.py only scores frames that appear in the file at all,
so leaving the remainder unlabelled would silently make the whole rest of
the clip NOT COMPUTED instead of counted as the "safe" evidence it is.
If you don't want that assumption (e.g. you only trust your own judgement
on part of the clip), pass --no-fill-safe and hand-edit the CSV afterwards.

This is intentionally a manual, subjective judgement call by whoever is
running the script -- see data/ground_truth/SCHEMA.md for what "unsafe"
is supposed to mean.
"""

import argparse
import csv
import os

import cv2


def annotate(video_path, video_source, out_path, fill_safe=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: could not open {video_path}")
        return 1

    unsafe_ranges = []  # list of (start_frame, end_frame), 1-indexed inclusive
    open_start = None
    frame_index = 0
    playing = True
    last_frame = None

    print(__doc__)

    while True:
        if playing or last_frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            last_frame = frame
        else:
            frame = last_frame

        display = frame.copy()
        status = "UNSAFE (open)" if open_start is not None else "safe"
        cv2.putText(display, f"frame {frame_index}  [{status}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("annotate ground truth", display)

        key = cv2.waitKey(30 if playing else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("n") and not playing:
            ok, frame = cap.read()
            if ok:
                frame_index += 1
                last_frame = frame
        elif key == ord("u"):
            if open_start is None:
                open_start = frame_index
                print(f"  unsafe range opened at frame {frame_index}")
            else:
                unsafe_ranges.append((open_start, frame_index))
                print(f"  unsafe range closed: {open_start}-{frame_index}")
                open_start = None

    if open_start is not None:
        unsafe_ranges.append((open_start, frame_index))
        print(f"  unsafe range closed at quit: {open_start}-{frame_index}")

    cap.release()
    cv2.destroyAllWindows()

    unsafe_ranges.sort()
    rows = [
        {"video_source": video_source, "start_frame": s, "end_frame": e, "label": "unsafe"}
        for s, e in unsafe_ranges
    ]

    if fill_safe:
        rows.extend(_safe_gaps(video_source, unsafe_ranges, frame_index))

    _write_csv(out_path, rows)
    print(f"wrote {len(rows)} row(s) to {out_path}")
    return 0


def _safe_gaps(video_source, unsafe_ranges, last_frame_index):
    """Every frame from 1..last_frame_index not covered by an unsafe range,
    collapsed into as few "safe" rows as possible."""
    gaps = []
    cursor = 1
    for start, end in unsafe_ranges:
        if cursor < start:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= last_frame_index:
        gaps.append((cursor, last_frame_index))
    return [
        {"video_source": video_source, "start_frame": s, "end_frame": e, "label": "safe"}
        for s, e in gaps
    ]


def _write_csv(out_path, rows):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Overwrite, not append -- re-running annotation on the same clip should
    # replace its previous ranges, not duplicate them. If multiple clips
    # share one file, merge by hand or extend this script; keeping it this
    # simple is deliberate (see module docstring).
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_source", "start_frame", "end_frame", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="path to the video file to annotate")
    parser.add_argument("--video-source", default=None,
                         help="value to write in the video_source column (default: the video path as given)")
    parser.add_argument("--out", default="data/ground_truth/frame_labels.csv",
                         help="output CSV path (default: data/ground_truth/frame_labels.csv)")
    parser.add_argument("--no-fill-safe", action="store_true",
                         help="do not auto-fill unlabelled frames as safe (see module docstring)")
    args = parser.parse_args()

    video_source = args.video_source if args.video_source is not None else args.video
    raise SystemExit(annotate(args.video, video_source, args.out, fill_safe=not args.no_fill_safe))


if __name__ == "__main__":
    main()
