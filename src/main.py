"""
Phase 0 scaffold, now with Phase 2's detector, Phase 5's tracker and
Phase 6's decision FSM wired in.

Opens either a video file (given as an argument) or the default webcam,
shows the frames, and quits on 'q'. When --weights is not given this is
still exactly the Phase 0 plain capture loop. When --weights is given,
each frame is run through src/detector.py, fed to a VehicleTracker
(Phase 5) and a CrossingFSM (Phase 6), and boxes + TTC + the current
verdict are drawn on it.
"""

import argparse
import os
import time

import cv2

from detector import Detector
from safety import VehicleTracker, APPROACHING, UNKNOWN
from fsm import CrossingFSM, SAFE_TO_CROSS, WAITING
from audio import AudioAnnouncer
from decision_log import DecisionLogger, DEFAULT_DB_PATH


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


def draw_detections(frame, detections, tracker=None):
    """Draw one box + label per detection directly onto the given frame.

    When a VehicleTracker is given (Phase 5), also draw that vehicle's
    current time-to-contact above its label, and the frame's min_ttc in a
    corner -- the "done" check for Phase 5 is that this number appears and
    visibly falls as a vehicle approaches.

    A vehicle's status (see safety.py: APPROACHING / NOT_APPROACHING /
    UNKNOWN) also shows up here, subtly, mainly so it's visible while
    debugging: APPROACHING gets the TTC number, UNKNOWN gets a small "?"
    (we cannot yet say whether it's a threat -- this is the one Phase 6
    must never read as "safe"), and NOT_APPROACHING gets nothing extra --
    an assessed, non-threatening vehicle needs no further comment.
    """
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

        if tracker is None:
            continue
        status = tracker.get_status(det.track_id)
        tag_y = max(label_y1 - 4, 10)

        if status == APPROACHING:
            ttc = tracker.get_ttc(det.track_id)
            # Red, above the class label -- a second line so it never overlaps it.
            cv2.putText(
                frame, f"TTC {ttc:.1f}s", (p1[0] + 2, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
        elif status == UNKNOWN:
            # Yellow "?" -- not enough data yet to say either way. Kept
            # deliberately small: this is a debugging aid, not a warning.
            cv2.putText(
                frame, "?", (p1[0] + 2, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2,
            )
        # status == NOT_APPROACHING (or no status yet): no extra text --
        # an assessed, non-threatening vehicle doesn't need a marker.


def draw_min_ttc(frame, min_ttc):
    """Draw the smallest TTC across the whole frame in the top-left corner."""
    text = f"min TTC: {min_ttc:.1f}s" if min_ttc is not None else "min TTC: --"
    color = (0, 0, 255) if min_ttc is not None else (200, 200, 200)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_verdict(frame, verdict):
    """Draw Phase 6's current state + reason, and the crosswalk direction
    when one is known. This is the Phase 6 "done" check from PLAN.md: the
    state and reason must be visible on the video and behave sanely.

    Colour follows the same "default to caution" rule the FSM itself
    follows: only SAFE_TO_CROSS gets green, everything else (including
    SEARCHING) is drawn in a caution colour, never green by default.
    """
    color = (0, 200, 0) if verdict.state == SAFE_TO_CROSS else (0, 165, 255)
    text = f"{verdict.state}: {verdict.reason}"
    cv2.putText(frame, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if verdict.crosswalk_direction is not None:
        cv2.putText(
            frame, f"crossing: {verdict.crosswalk_direction}", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )


def run(cap, headless, max_frames, detector, save_frames_dir=None, save_frames_every=15,
        announcer=None, decision_logger=None, video_source=None):
    """Read frames until the source ends, max_frames is hit, or 'q' is pressed.

    When detector is None this is the plain Phase 0 loop, unchanged. When a
    detector is given, each frame is also run through it, fed to a
    VehicleTracker (Phase 5) for time-to-contact and a CrossingFSM
    (Phase 6) for the crossing verdict, and boxes + TTC + the verdict are
    drawn.

    save_frames_dir, when given, writes an annotated JPEG every
    save_frames_every frames (capped at 3 total) -- used to produce the
    Phase 6 "done" check screenshots without touching the report/
    generation scripts under scripts/.

    announcer, when given (Phase 7, requires detector), gets every frame's
    verdict handed to it. AudioAnnouncer itself decides whether a given
    frame is worth speaking -- see audio.py -- and never blocks this loop:
    speech runs on its own thread, and announce() never raises.

    decision_logger, when given (Phase 8, requires detector), gets every
    frame's verdict + detections handed to it for logs/decisions.db. Like
    announcer, it never blocks this loop beyond a queue put and never
    raises -- see decision_log.py.
    """
    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    inference_times = []  # seconds per frame, only filled in when detector is set
    saved_frames = 0

    # Only needed when we actually have detections to track/decide on. Real
    # elapsed time (time.time()), not a frame counter, feeds safety.py's dt
    # -- the pipeline's fps varies (8-15 fps), so a fixed-dt assumption
    # would make the TTC estimate wrong.
    tracker = VehicleTracker() if detector is not None else None
    fsm = CrossingFSM() if detector is not None else None

    if save_frames_dir is not None:
        os.makedirs(save_frames_dir, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            # end of video file, or camera stopped giving frames
            break

        frame_count += 1

        if detector is not None:
            frame_start = time.time()
            infer_start = frame_start
            detections = detector.detect(frame)
            inference_s = time.time() - infer_start
            inference_times.append(inference_s)
            tracker.update(detections, infer_start)
            verdict = fsm.update(detections, tracker, width)
            draw_detections(frame, detections, tracker)
            draw_min_ttc(frame, tracker.min_ttc())
            draw_verdict(frame, verdict)

            if announcer is not None:
                announcer.announce(verdict)

            if decision_logger is not None:
                # total_ms covers frame capture through here -- the point
                # audio.announce() has (or hasn't) already been kicked off
                # above -- which is PLAN.md's "capture to audio start"
                # latency. announce() itself only ever enqueues (see
                # audio.py), so this is measured, not merely assumed, to be
                # representative of that latency.
                total_ms = 1000.0 * (time.time() - frame_start)
                decision_logger.log(
                    frame_count,
                    verdict,
                    tracker.unresolved_vehicles(),
                    detections,
                    inference_ms=1000.0 * inference_s,
                    total_ms=total_ms,
                )

            if (
                save_frames_dir is not None
                and saved_frames < 3
                and frame_count % save_frames_every == 0
            ):
                out_path = os.path.join(save_frames_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_frames += 1

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
    parser.add_argument(
        "--save-frames",
        default=None,
        metavar="DIR",
        help="save up to 3 annotated frames (state + reason drawn) to this "
        "directory, for phase done-check screenshots; requires --weights",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="disable Phase 7 audio (speech + earcons) even when --weights is given; "
        "useful for testing the video path silently",
    )
    parser.add_argument(
        "--log-db",
        default=DEFAULT_DB_PATH,
        metavar="PATH",
        help=f"Phase 8 decision-log SQLite path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="disable Phase 8 decision logging even when --weights is given",
    )
    args = parser.parse_args()

    cap, is_camera = open_source(args.video)

    detector = Detector(args.weights) if args.weights else None
    # Audio only makes sense once there's a verdict to announce, which needs
    # a detector -- and only when the caller hasn't asked for --mute. This
    # keeps --headless + audio a supported combination on its own (the
    # window is optional, the announcer is not tied to it).
    announcer = AudioAnnouncer() if (detector is not None and not args.mute) else None
    # Same shape as the announcer: only makes sense with a detector (there
    # is no verdict to log without one), and only when not explicitly
    # disabled. A construction failure (bad path, disk full, ...) degrades
    # to a warning inside DecisionLogger itself, never a crash here.
    video_source_name = args.video if args.video is not None else "camera 0"
    decision_logger = (
        DecisionLogger(db_path=args.log_db, video_source=video_source_name)
        if (detector is not None and not args.no_log)
        else None
    )

    max_frames = args.max_frames
    # A live camera never hits end-of-stream on its own. Without a display
    # there is no 'q' key either, so --headless on the camera would run
    # forever unless we cap it. A file source has a real EOF, so leave it
    # alone unless the user asked for a cap explicitly.
    if args.headless and is_camera and max_frames is None:
        max_frames = 300
        print("note: headless camera capture, stopping after 300 frames (override with --max-frames)")

    try:
        run(
            cap,
            args.headless,
            max_frames,
            detector,
            save_frames_dir=args.save_frames,
            announcer=announcer,
            decision_logger=decision_logger,
            video_source=video_source_name,
        )
    finally:
        cap.release()
        if not args.headless:
            # Only a windowed run has windows to destroy. Calling this on an
            # OpenCV build without GUI support raises, which would turn a
            # clean headless run into a crash for no reason.
            cv2.destroyAllWindows()
        if announcer is not None:
            # Prompt, bounded shutdown -- see audio.py's close(): a daemon
            # thread plus a joined sentinel, so this can never hang exit.
            announcer.close()
        if decision_logger is not None:
            # Same bounded-shutdown shape -- flush the writer thread's
            # remaining batch and stop it, without risking a hang on exit.
            decision_logger.close()


if __name__ == "__main__":
    main()
