"""
Phase 0 scaffold: just a capture loop.

Opens either a video file (given as an argument) or the default webcam,
shows the frames, and quits on 'q'. Later phases will plug a detector into
the run() loop without needing to touch the open/close logic here.
"""

import argparse
import time

import cv2


def open_source(source_arg):
    """Open a video file if a path was given, otherwise the default camera.

    Returns (cap, is_camera) so the caller knows which kind of source it got.
    """
    if source_arg is None:
        cap = cv2.VideoCapture(0)
        source_name = "camera 0"
        is_camera = True
    else:
        cap = cv2.VideoCapture(source_arg)
        source_name = source_arg
        is_camera = False

    if not cap.isOpened():
        print(f"error: could not open video source: {source_name}")
        raise SystemExit(1)

    return cap, is_camera


def run(cap, headless, max_frames):
    """Read frames until the source ends, max_frames is hit, or 'q' is pressed."""
    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            # end of video file, or camera stopped giving frames
            break

        frame_count += 1

        if not headless:
            cv2.imshow("road-crossing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if max_frames is not None and frame_count >= max_frames:
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    if headless:
        print(f"frames read: {frame_count}")
        print(f"resolution: {width}x{height}")
        print(f"measured fps: {fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Road-crossing capture loop (Phase 0)")
    parser.add_argument("video", nargs="?", default=None, help="path to a video file (default: webcam 0)")
    parser.add_argument("--headless", action="store_true", help="run without a display window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    args = parser.parse_args()

    cap, is_camera = open_source(args.video)

    max_frames = args.max_frames
    # A live camera never hits end-of-stream on its own. Without a display
    # there is no 'q' key either, so --headless on the camera would run
    # forever unless we cap it. A file source has a real EOF, so leave it
    # alone unless the user asked for a cap explicitly.
    if args.headless and is_camera and max_frames is None:
        max_frames = 300
        print("note: headless camera capture, stopping after 300 frames (override with --max-frames)")

    try:
        run(cap, args.headless, max_frames)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
