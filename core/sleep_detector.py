import cv2
import numpy as np
import mediapipe as mp
import time
from config import EAR_THRESHOLD, SLEEP_HOLD_SEC
from config import SLEEP_HEAD_DELTA, SLEEP_HEAD_HOLD

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    refine_landmarks=False
)


def eye_aspect_ratio(landmarks, indices):
    try:
        p  = [(landmarks[i].x, landmarks[i].y)
              for i in indices]
        v1 = ((p[1][0]-p[5][0])**2 +
               (p[1][1]-p[5][1])**2) ** 0.5
        v2 = ((p[2][0]-p[4][0])**2 +
               (p[2][1]-p[4][1])**2) ** 0.5
        h  = ((p[0][0]-p[3][0])**2 +
               (p[0][1]-p[3][1])**2) ** 0.5
        return (v1 + v2) / (2.0 * h + 1e-6)
    except:
        return None


def get_face_center(landmarks, frame_w, frame_h):
    try:
        nose = landmarks[1]
        return (
            int(nose.x * frame_w),
            int(nose.y * frame_h)
        )
    except:
        return None


def bbox_contains_point(bbox, point, margin=40):
    x1, y1, x2, y2 = bbox
    px, py = point
    return (x1 - margin <= px <= x2 + margin and
            y1 - margin <= py <= y2 + margin)


def get_all_ears(frame):
    h, w = frame.shape[:2]

    try:
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(
            clipLimit=3.0, tileGridSize=(8, 8)
        )
        l        = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(
            enhanced, cv2.COLOR_LAB2BGR
        )
    except:
        enhanced = frame

    faces = []

    for test_frame in [enhanced, frame]:
        rgb     = cv2.cvtColor(
            test_frame, cv2.COLOR_BGR2RGB
        )
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        for face_lm in results.multi_face_landmarks:
            lm        = face_lm.landmark
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
            center    = get_face_center(lm, w, h)

            if left_ear and right_ear and center:
                ear = (left_ear + right_ear) / 2.0
                faces.append((center, ear))

        if faces:
            break

    return faces


def head_down_score(kpts, bbox):
    try:
        nose = kpts[0]
        ls   = kpts[5]
        rs   = kpts[6]
        if nose[0] <= 0 or ls[0] <= 0 or rs[0] <= 0:
            return None
        shoulder_y      = (ls[1] + rs[1]) / 2.0
        x1, y1, x2, y2 = bbox
        h               = max(1, y2 - y1)
        return float((nose[1] - shoulder_y) / h)
    except:
        return None


class SleepDetector:
    """
    PRIMARY:   Head drop (reliable always)
    SECONDARY: EAR (backup when face visible)
    """

    def __init__(self):
        self.sleep_start      = None
        self.head_sleep_start = None
        self.is_sleeping      = False
        self.ear_history      = []
        self.smooth_size      = 5
        self.last_ear         = 0.4
        self.baseline         = None
        self.baseline_samples = []

    def _smooth_ear(self, ear):
        self.ear_history.append(ear)
        if len(self.ear_history) > self.smooth_size:
            self.ear_history.pop(0)
        return float(np.mean(self.ear_history))

    def _build_baseline(self, dscore):
        if self.baseline is not None:
            return
        self.baseline_samples.append(dscore)
        if len(self.baseline_samples) >= 18:
            self.baseline = float(
                np.median(self.baseline_samples)
            )

    def update(self, frame, bbox, kpts,
               all_ears, now):

        if bbox is None:
            return False, self.last_ear

        # ── PRIMARY: HEAD DROP ───────────────────────────────────
        head_sleeping = False
        if kpts is not None:
            dscore = head_down_score(kpts, bbox)
            if dscore is not None:
                self._build_baseline(dscore)
                if self.baseline is not None:
                    delta = dscore - self.baseline
                    if delta > SLEEP_HEAD_DELTA:
                        if self.head_sleep_start is None:
                            self.head_sleep_start = now
                        if (now - self.head_sleep_start
                                >= SLEEP_HEAD_HOLD):
                            head_sleeping = True
                    else:
                        self.head_sleep_start = None

        # ── SECONDARY: EAR ───────────────────────────────────────
        ear_sleeping = False
        ear          = None

        for (fx, fy), face_ear in all_ears:
            if bbox_contains_point(bbox, (fx, fy)):
                ear = face_ear
                break

        if ear is None:
            ear = self.last_ear
        else:
            self.last_ear = ear

        smooth_ear = self._smooth_ear(ear)

        if smooth_ear < EAR_THRESHOLD:
            ear_sleeping = True

        # ── COMBINE ──────────────────────────────────────────────
        is_drowsy = head_sleeping or ear_sleeping

        if is_drowsy:
            if self.sleep_start is None:
                self.sleep_start = now
        else:
            self.sleep_start = None
            self.is_sleeping = False

        if (self.sleep_start is not None and
                (now - self.sleep_start) >= SLEEP_HOLD_SEC):
            self.is_sleeping = True
        else:
            if not is_drowsy:
                self.is_sleeping = False

        return self.is_sleeping, smooth_ear