import time
import numpy as np
from config import (
    BASELINE_MIN_SAMPLES,
    BASELINE_SIDE_MAX,
    DOWN_DELTA,
    DOWN_HOLD_SEC,
    SLEEP_HEAD_DELTA,
    SLEEP_HEAD_HOLD,
)

# ── Keypoint indices ─────────────────────────────────────────────
NOSE           = 0
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12

# ── TALKING config ───────────────────────────────────────────────
TALKING_HOLD_SEC     = 6.0   # DISTRACTED → TALKING
TALKING_SIDE_FRAMES  = 15    # frames to confirm side direction
TALKING_DIR_MEMORY   = 30    # frames to remember last direction


def head_down_score(kpts, bbox):
    try:
        nose = kpts[NOSE]
        ls   = kpts[LEFT_SHOULDER]
        rs   = kpts[RIGHT_SHOULDER]
        if nose[0] <= 0 or ls[0] <= 0 or rs[0] <= 0:
            return None
        shoulder_y      = (ls[1] + rs[1]) / 2.0
        x1, y1, x2, y2 = bbox
        h               = max(1, y2 - y1)
        return float((nose[1] - shoulder_y) / h)
    except:
        return None


def side_score(kpts):
    try:
        nose = kpts[NOSE]
        ls   = kpts[LEFT_SHOULDER]
        rs   = kpts[RIGHT_SHOULDER]
        if nose[0] <= 0 or ls[0] <= 0 or rs[0] <= 0:
            return None
        shoulder_cx = (ls[0] + rs[0]) / 2.0
        shoulder_w  = abs(rs[0] - ls[0])
        if shoulder_w < 2:
            return None
        return float(
            abs(nose[0] - shoulder_cx) / shoulder_w
        )
    except:
        return None


def nose_shoulder_ratio(kpts):
    try:
        nose = kpts[NOSE]
        ls   = kpts[LEFT_SHOULDER]
        rs   = kpts[RIGHT_SHOULDER]
        if nose[0] <= 0 or ls[0] <= 0 or rs[0] <= 0:
            return None
        dist_left  = ((nose[0] - ls[0])**2 +
                      (nose[1] - ls[1])**2) ** 0.5
        dist_right = ((nose[0] - rs[0])**2 +
                      (nose[1] - rs[1])**2) ** 0.5
        if dist_right < 0.001:
            return 1.0
        return float(dist_left / dist_right)
    except:
        return None


def head_movement_score(nose_history):
    if len(nose_history) < 5:
        return 1.0
    positions = list(nose_history)
    movements = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        movements.append((dx**2 + dy**2) ** 0.5)
    return float(np.mean(movements))


class PersonPoseState:

    def __init__(self):
        self.baseline_samples   = []
        self.baseline           = None
        self.nose_history       = []
        self.nose_history_size  = 20
        self.right_counter      = 0
        self.left_counter       = 0
        self.consecutive_frames = 15
        self.sleep_start        = None
        self.head_down_start    = None
        self.pose_label         = "CALIBRATING"

        # ── TALKING state variables ──────────────────────────────
        self.distracted_start    = None   # when distraction began
        self.distracted_dir      = None   # "LEFT" or "RIGHT"
        self.dir_change_count    = 0      # how many times direction changed
        self.talking_confirmed   = False

    def is_calibrated(self):
        return self.baseline is not None

    def _update_nose_history(self, kpts):
        try:
            nose = kpts[NOSE]
            if nose[0] > 0 and nose[1] > 0:
                self.nose_history.append(
                    (float(nose[0]), float(nose[1]))
                )
                if len(self.nose_history) > \
                        self.nose_history_size:
                    self.nose_history.pop(0)
        except:
            pass

    def _build_baseline(self, dscore, sscore):
        if sscore <= BASELINE_SIDE_MAX:
            self.baseline_samples.append(dscore)
        if len(self.baseline_samples) >= \
                BASELINE_MIN_SAMPLES:
            self.baseline = float(
                np.median(self.baseline_samples)
            )
            self.pose_label = "ENGAGED"

    def _update_talking(self, ratio, is_distracted, now):
        """
        TALKING detection logic:
        - Student side dekhe (DISTRACTED)
        - Same side continuously for TALKING_HOLD_SEC
        - Direction change hone par TALKING cancel
        """

        if ratio is None:
            return False

        # Current direction
        if ratio > 1.45:
            current_dir = "RIGHT"
        elif ratio < 0.68:
            current_dir = "LEFT"
        else:
            current_dir = None  # forward

        # Student forward aa gaya — reset
        if current_dir is None:
            self.distracted_start  = None
            self.distracted_dir    = None
            self.dir_change_count  = 0
            self.talking_confirmed = False
            return False

        # Is distracted sideways
        if is_distracted:

            # Direction change detect
            if (self.distracted_dir is not None and
                    current_dir != self.distracted_dir):
                # Direction badal gaya — TALKING nahi
                # Reset timer
                self.dir_change_count += 1
                self.distracted_start  = now
                self.distracted_dir    = current_dir
                self.talking_confirmed = False
                return False

            # Same direction — start/continue timer
            if self.distracted_dir is None:
                self.distracted_dir   = current_dir
                self.distracted_start = now

            # Check time elapsed same side
            elapsed = now - self.distracted_start

            if elapsed >= TALKING_HOLD_SEC:
                # Only confirm if direction never changed
                if self.dir_change_count == 0:
                    self.talking_confirmed = True
                    return True

        else:
            # Not distracted — reset talking
            self.distracted_start  = None
            self.distracted_dir    = None
            self.dir_change_count  = 0
            self.talking_confirmed = False

        return self.talking_confirmed

    def update(self, kpts, bbox, now):
        if kpts is None:
            return self.pose_label

        dscore = head_down_score(kpts, bbox)
        sscore = side_score(kpts)
        ratio  = nose_shoulder_ratio(kpts)

        if dscore is None or sscore is None:
            return self.pose_label

        self._update_nose_history(kpts)

        # ── CALIBRATION ──────────────────────────────────────────
        if self.baseline is None:
            self._build_baseline(dscore, sscore)
            return "CALIBRATING"

        delta = dscore - self.baseline

        # ── DISTRACTION (side) ───────────────────────────────────
        if ratio is not None:
            if ratio > 1.45:
                self.right_counter += 1
                self.left_counter   = 0
            elif ratio < 0.68:
                self.left_counter  += 1
                self.right_counter  = 0
            else:
                self.right_counter = max(
                    0, self.right_counter - 2
                )
                self.left_counter  = max(
                    0, self.left_counter - 2
                )

        is_distracted = (
            self.right_counter >= self.consecutive_frames
            or
            self.left_counter  >= self.consecutive_frames
        )

        # ── TALKING check ────────────────────────────────────────
        is_talking = self._update_talking(
            ratio, is_distracted, now
        )

        # ── SLEEPING ─────────────────────────────────────────────
        movement   = head_movement_score(self.nose_history)
        head_down  = delta > SLEEP_HEAD_DELTA
        head_still = movement < 5.0

        is_sleeping_signal = head_down and head_still

        if is_sleeping_signal:
            if self.sleep_start is None:
                self.sleep_start = now
        else:
            self.sleep_start = None

        sleep_confirmed = (
            self.sleep_start is not None and
            (now - self.sleep_start) >= SLEEP_HEAD_HOLD
        )

        # ── HEAD DOWN TIMER ──────────────────────────────────────
        if delta > SLEEP_HEAD_DELTA:
            if self.head_down_start is None:
                self.head_down_start = now
        else:
            self.head_down_start = None

        head_down_duration = (
            (now - self.head_down_start)
            if self.head_down_start is not None
            else 0.0
        )

        # ── PRIORITY ─────────────────────────────────────────────
        # SLEEPING > TALKING > DISTRACTED > ENGAGED
        if sleep_confirmed:
            self.pose_label = "SLEEPING"
        elif is_talking:
            self.pose_label = "TALKING"
        elif is_distracted:
            self.pose_label = "DISTRACTED"
        elif head_down_duration > DOWN_HOLD_SEC:
            self.pose_label = "DISTRACTED"
        else:
            self.pose_label = "ENGAGED"

        return self.pose_label