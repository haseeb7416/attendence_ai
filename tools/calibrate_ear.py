import cv2
import time
import numpy as np
import mediapipe as mp
from config import FRAME_WIDTH, FRAME_HEIGHT

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Use full frame face detection
# No cropping — most reliable
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("EAR CALIBRATION — Full Frame Mode")
print("=" * 40)
print("Face detection on full frame")
print("More reliable than crop mode")

def collect(instruction, seconds=5):
    print(f"\n>>> {instruction}")
    print(f"    Starting in 3 seconds...")
    time.sleep(3)
    print(f"    Collecting {seconds} seconds...")

    samples     = []
    start       = time.time()
    no_face_cnt = 0

    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm        = results.multi_face_landmarks[0].landmark
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE)

            if left_ear and right_ear:
                ear = (left_ear + right_ear) / 2.0
                samples.append(ear)

                color = (0, 255, 0)
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  samples:{len(samples)}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2
                )
        else:
            no_face_cnt += 1
            cv2.putText(
                frame,
                "NO FACE DETECTED - move closer",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.putText(
            frame, instruction[:35],
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 0), 1
        )

        remaining = seconds - (time.time() - start)
        cv2.putText(
            frame,
            f"Time left: {remaining:.1f}s",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (200, 200, 200), 1
        )

        cv2.imshow("EAR Calibration", frame)
        cv2.waitKey(1)

    print(f"    Collected {len(samples)} samples")
    print(f"    No face frames: {no_face_cnt}")
    return samples

open_samples   = collect("Keep eyes FULLY OPEN", 5)
closed_samples = collect("CLOSE eyes completely", 5)

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 40)

if open_samples and closed_samples:
    open_mean   = np.mean(open_samples)
    closed_mean = np.mean(closed_samples)
    gap         = open_mean - closed_mean
    threshold   = (open_mean + closed_mean) / 2.0

    print(f"Eyes OPEN   mean EAR : {open_mean:.3f}")
    print(f"Eyes CLOSED mean EAR : {closed_mean:.3f}")
    print(f"Gap                  : {gap:.3f}")
    print(f"Recommended threshold: {round(threshold, 3)}")
    print("=" * 40)
    print(f"\nUpdate config.py:")
    print(f"EAR_THRESHOLD = {round(threshold, 3)}")

    if gap < 0.05:
        print("\nWARNING: Gap too small!")
        print("Camera or lighting may be causing issues")
        print("Try better lighting or move closer")
else:
    print("No samples collected!")
    print("Make sure:")
    print("  1. Your face is clearly visible")
    print("  2. Good lighting on your face")
    print("  3. Camera is at eye level")