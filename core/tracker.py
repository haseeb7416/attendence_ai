import math
import numpy as np
import cv2
from ultralytics import YOLO
from config import (
    POSE_MODEL, DET_MODEL,
    YOLO_IMGSZ, POSE_EVERY_N, DET_EVERY_N
)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1  = max(ax1, bx1)
    iy1  = max(ay1, by1)
    ix2  = min(ax2, bx2)
    iy2  = min(ay2, by2)
    iw   = max(0, ix2 - ix1)
    ih   = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)


class TrackedPerson:
    def __init__(self, track_id, bbox, kpts=None):
        self.track_id  = track_id
        self.bbox      = bbox
        self.kpts      = kpts
        self.student_id   = None
        self.student_name = None


class YOLOByteTracker:
    """
    Professional tracker using YOLO built-in ByteTrack.
    Handles:
        - Free movement
        - Overlapping
        - Re-entry
        - Multi person
    """

    def __init__(self):
        print("Loading YOLO pose model...")
        self.pose_model = YOLO(POSE_MODEL)
        print("Loading YOLO detection model...")
        self.det_model  = YOLO(DET_MODEL)
        print("Models loaded successfully!")

        self.frame_idx         = 0
        self.last_person_boxes = []
        self.last_results      = []

    def update(self, frame):
        self.frame_idx += 1
        tracked_persons = []

        # ── Run pose model with ByteTrack ───────────────────────
        if self.frame_idx % POSE_EVERY_N == 0:
            results = self.pose_model.track(
                frame,
                conf=0.45,
                imgsz=YOLO_IMGSZ,
                classes=[0],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )
            self.last_results = results

        # ── Run detection model for person validation ───────────
        if self.frame_idx % DET_EVERY_N == 0:
            det = self.det_model.predict(
                frame,
                conf=0.60,
                imgsz=YOLO_IMGSZ,
                classes=[0],
                verbose=False
            )[0]
            self.last_person_boxes = []
            if det.boxes is not None:
                for b in det.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, b)
                    self.last_person_boxes.append(
                        (x1, y1, x2, y2)
                    )

        # ── Process ByteTrack results ───────────────────────────
        if not self.last_results:
            return tracked_persons

        result = self.last_results[0]

        if result.boxes is None:
            return tracked_persons

        if result.boxes.id is None:
            return tracked_persons

        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        kpts_all  = (
            result.keypoints.xy.cpu().numpy()
            if result.keypoints is not None
            else None
        )

        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            bbox = (x1, y1, x2, y2)

            # Validate against person detection
            valid = False
            if self.last_person_boxes:
                for pb in self.last_person_boxes:
                    if iou(bbox, pb) >= 0.25:
                        valid = True
                        break
            else:
                valid = True

            if not valid:
                continue

            # Filter small detections (furniture/sofa)
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            min_h  = frame.shape[0] * 0.15
            min_w  = frame.shape[1] * 0.05
            if bbox_h < min_h or bbox_w < min_w:
                continue

            # Get keypoints
            kpts = kpts_all[i] if kpts_all is not None else None

            # Filter non-humans by checking keypoints
            # Real person has valid nose + shoulders
            # Sofa/furniture has no valid keypoints
            if kpts is not None:
                nose = kpts[0]
                ls   = kpts[5]
                rs   = kpts[6]
                has_nose      = nose[0] > 0 and nose[1] > 0
                has_shoulders = (
                    (ls[0] > 0 and ls[1] > 0) or
                    (rs[0] > 0 and rs[1] > 0)
                )
                if not has_nose and not has_shoulders:
                    continue

            person = TrackedPerson(
                track_id=tid,
                bbox=bbox,
                kpts=kpts
            )
            tracked_persons.append(person)

        return tracked_persons