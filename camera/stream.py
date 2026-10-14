import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_attendance.config import CAMERA_SOURCE

def get_camera():
    cap = cv2.VideoCapture(
    CAMERA_SOURCE, cv2.CAP_FFMPEG
)
    if not cap.isOpened():
        print("Cannot open camera!")
        return None
    print("Camera opened successfully!")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()