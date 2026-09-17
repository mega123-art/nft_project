"""
Phase 2 step 4: YOLO detector wrapper.

Wraps an Ultralytics model and turns its raw output into a small typed
Detection list, so the rest of the codebase (main.py now, safety.py/fsm.py
later) never has to touch Ultralytics' Results objects directly.
"""

from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int]  # None until the tracker has assigned this box an id


class Detector:
    def __init__(self, weights_path, conf=0.4):
        self.model = YOLO(weights_path)
        self.conf = conf
        # class names come from the model itself (whatever it was trained
        # on -- COCO's 80 classes for stock yolov8n.pt, our 9 classes once
        # Phase 2 training is done), never a hardcoded list here.
        self.class_names = self.model.names

    def detect(self, frame):
        """Run detection + tracking on one frame, return a list of Detection."""
        # persist=True is critical: it keeps the tracker's state between
        # calls so the same vehicle keeps the same track_id across frames.
        # Without it every call starts a fresh tracker, ids reset to 1, 2, 3...
        # each frame, and Phase 5's time-to-contact (which needs a track_id's
        # box width over several frames) becomes garbage. Do not remove this.
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)

        detections = []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            # box.id is None when the tracker hasn't assigned this detection
            # an id yet (e.g. very first frame it appears in) -- handle that
            # instead of crashing on .item().
            track_id = int(box.id[0]) if box.id is not None else None

            detections.append(
                Detection(
                    cls_name=cls_name,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    track_id=track_id,
                )
            )

        return detections
