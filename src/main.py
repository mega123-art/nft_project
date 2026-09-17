"""
Phase 0 scaffold, now with Phase 2's detector wired in.

Opens either a video file (given as an argument) or the default webcam,
shows the frames, and quits on 'q'. When --weights is not given this is
still exactly the Phase 0 plain capture loop. When --weights is given,
each frame is run through src/detector.py and boxes are drawn on it.
"""

import argparse
import time

import cv2

from detector import Detector


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


def draw_detections(frame, detections):
    """Draw one box + label per detection directly onto the given frame."""
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        id_part = f" id{det.track_id}" if det.track_id is not None else ""
        label = f"{det.cls_name} {det.conf:.2f}{id_part}"

        # small filled bar behind the text so it stays readable over any
        # background colour, instead of raw text with no contrast
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(p1[1] - text_h - 6, 0)
        cv2.rectangle(frame, (p1[0], label_y1), (p1[0] + text_w + 4, p1[1]), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )


def run(cap, headless, max_frames, detector):
    """Read frames until the source ends, max_frames is hit, or 'q' is pressed.

    When detector is None this is the plain Phase 0 loop, unchanged. When a
    detector is given, each frame is also run through it and boxes are drawn.
    """
    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    inference_times = []  # seconds per frame, only filled in when detector is set

    while True:
        ok, frame = cap.read()
        if not ok:
            # end of video file, or camera stopped giving frames
            break

        frame_count += 1

        if detector is not None:
            infer_start = time.time()
            detections = detector.detect(frame)
            inference_times.append(time.time() - infer_start)
            draw_detections(frame, detections)

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

        if inference_times:
            mean_ms = 1000.0 * sum(inference_times) / len(inference_times)
            inference_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            print(f"mean inference time: {mean_ms:.1f} ms/frame")
            print(f"inference-only fps: {inference_fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Road-crossing capture loop (Phase 0)")
    parser.add_argument("video", nargs="?", default=None, help="path to a video file (default: webcam 0)")
    parser.add_argument("--headless", action="store_true", help="run without a display window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    parser.add_argument(
        "--weights",
        default=None,
        help="path to YOLO weights; when given, run detection and draw boxes",
    )
    args = parser.parse_args()

    cap, is_camera = open_source(args.video)

    detector = Detector(args.weights) if args.weights else None

    max_frames = args.max_frames
    # A live camera never hits end-of-stream on its own. Without a display
    # there is no 'q' key either, so --headless on the camera would run
    # forever unless we cap it. A file source has a real EOF, so leave it
    # alone unless the user asked for a cap explicitly.
    if args.headless and is_camera and max_frames is None:
        max_frames = 300
        print("note: headless camera capture, stopping after 300 frames (override with --max-frames)")

    try:
        run(cap, args.headless, max_frames, detector)
    finally:
        cap.release()
        if not args.headless:
            # Only a windowed run has windows to destroy. Calling this on an
            # OpenCV build without GUI support raises, which would turn a
            # clean headless run into a crash for no reason.
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
