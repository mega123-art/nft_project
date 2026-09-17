"""
Phase 4 step 2: pull still frames out of our own footage for labelling.

Writes one JPG every --interval seconds (default 0.5s, per PLAN.md) into
data/frames/<video_stem>/. Frame filenames embed the source video's stem and
the original frame index, e.g. car-detection_f00042.jpg, so a frame can
always be traced back to the video it came from. PLAN.md is explicit about
why this matters: if our own footage is later split into train/val by frame
instead of by video, near-identical neighbouring frames end up on both sides
of the split and the validation score becomes meaningless. Keeping the video
identifiable in the filename is what lets a later splitting step group by
video correctly.

Uses OpenCV, not ffmpeg -- ffmpeg is not installed on this machine (see
PLAN.md's Phase 0 environment deviations).
"""

import argparse
import os

import cv2

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def find_videos(input_path):
    """A single video file, or every video file directly inside a directory."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        videos = []
        for name in sorted(os.listdir(input_path)):
            if name.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(input_path, name))
        return videos

    print(f"error: no such file or directory: {input_path}")
    raise SystemExit(1)


def extract_one(video_path, out_root, interval, max_frames):
    """Extract frames from one video, return the number of frames written."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"warning: could not open {video_path}, skipping")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # some containers report 0 fps; fall back to a sane default rather
        # than dividing by zero below
        print(f"warning: {video_path} reports no fps, assuming 25")
        fps = 25.0

    # how many source frames to advance between saved frames. Rounding
    # (rather than truncating) keeps the actual spacing closest to the
    # requested --interval on typical fps values like 12.5 or 29.97.
    step = max(1, round(interval * fps))

    frame_index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            out_name = f"{stem}_f{frame_index:05d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, frame)
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        frame_index += 1

    cap.release()
    print(f"{video_path}: read {frame_index} frames at {fps:.2f} fps, saved {saved} to {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from footage for labelling (Phase 4)")
    parser.add_argument("input", help="a video file, or a directory of videos")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between saved frames (default 0.5)")
    parser.add_argument("--out-dir", default="data/frames", help="root output directory (default data/frames)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many saved frames, per video")
    args = parser.parse_args()

    videos = find_videos(args.input)
    if not videos:
        print(f"no video files found under {args.input}")
        raise SystemExit(1)

    total = 0
    for video_path in videos:
        total += extract_one(video_path, args.out_dir, args.interval, args.max_frames)

    print(f"\ntotal frames saved: {total}")


if __name__ == "__main__":
    main()
