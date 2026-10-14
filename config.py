import os

# ── Base Paths ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ── Backend ──────────────────────────────────────────────────────
BASE_URL = "https://flyless-lavern-unstealthy.ngrok-free.dev"
CLASS_ID = "83ad2d5b-3153-4cdf-8600-6f295008b109"

# ── Models ───────────────────────────────────────────────────────
POSE_MODEL   = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
DET_MODEL    = os.path.join(MODELS_DIR, "yolov8n.pt")

# ── Camera 
#CAMERA_SOURCE = 0
CAMERA_SOURCE = "rtsp://admin:admin123456@192.168.111.3:554/ch=1&subtype=1"
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
YOLO_IMGSZ    = 320

# ── Detection Intervals ──────────────────────────────────────────
POSE_EVERY_N  = 3
DET_EVERY_N   = 4
FACE_EVERY_N  = 30

# ── Tracker ──────────────────────────────────────────────────────
TRACKER_MAX_DISTANCE = 200
TRACKER_MIN_IOU      = 0.10
TRACKER_TTL          = 35
TRACKER_HIDE_MISSED  = 8
TRACKER_DROP_MISSED  = 18

# ── Baseline Calibration ─────────────────────────────────────────
BASELINE_MIN_SAMPLES = 18
BASELINE_SIDE_MAX    = 0.18

# ── Distraction ──────────────────────────────────────────────────
DOWN_DELTA    = 0.05
DOWN_HOLD_SEC = 3.0
SIDE_THRESH   = 0.16
SIDE_HOLD_SEC = 1.2


# ── Sleeping ─────────────────────────────────────────────────────
EAR_THRESHOLD = 0.40
SLEEP_HOLD_SEC = 2.5
SLEEP_HEAD_DELTA = 0.06
SLEEP_HEAD_HOLD  = 10.0
DOWN_HOLD_SEC    = 3.0


# ── Talking ──────────────────────────────────────────────────────
TALKING_HOLD_SEC = 6.0


# ── Face Recognition ─────────────────────────────────────────────
FACE_MODEL     = "SFace"
FACE_THRESHOLD = 0.45

# ── Session ───────────────────────────────────────────────────────
SESSION_POLL_SEC = 5

# ── States ───────────────────────────────────────────────────────
STATES = [
    "ENGAGED",
    "DISTRACTED",
    "SLEEPING",
    "TALKING"
]