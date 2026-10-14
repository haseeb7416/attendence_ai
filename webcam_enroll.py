import cv2
import requests
import time
import json
import numpy as np
from deepface import DeepFace
from config import BASE_URL, CLASS_ID, FACE_MODEL, CAMERA_SOURCE

print("=" * 55)
print("ENROLLMENT TOOL")
print("=" * 55)
print("This tool captures student photos")
print("from Tenda camera for recognition")
print("=" * 55)


def get_students(class_id):
    try:
        r = requests.get(
            f"{BASE_URL}/students?class_id={class_id}",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=10
        )
        return r.json()
    except Exception as e:
        print(f"Error getting students: {e}")
        return []


def save_all_embeddings(student_id, embeddings):
    """Save all embeddings using new endpoint"""
    for attempt in range(3):
        try:
            r = requests.post(
                f"{BASE_URL}/students/{student_id}/embeddings",
                json={"face_embeddings": embeddings},
                headers={
                    "ngrok-skip-browser-warning": "true"
                },
                timeout=30
            )
            return r.json()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def extract_embedding(frame):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result  = DeepFace.represent(
            img_path=img_rgb,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend="opencv"
        )
        return result[0]["embedding"]
    except:
        return None


def capture_photos_for_student(student, cap):
    name       = student["full_name"]
    student_id = student["id"]

    print(f"\n{'='*55}")
    print(f"Enrolling: {name}")
    print(f"{'='*55}")

    instructions = [
        "Look STRAIGHT at camera",
        "Turn head SLIGHTLY LEFT",
        "Turn head SLIGHTLY RIGHT",
        "Look SLIGHTLY UP",
        "Look SLIGHTLY DOWN"
    ]

    embeddings = []

    for i, instruction in enumerate(instructions):
        print(f"\nPhoto {i+1}/5: {instruction}")
        print("Press SPACE to capture, Q to skip")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            cv2.rectangle(
                display, (0, 0),
                (display.shape[1], 60),
                (0, 0, 0), -1
            )
            cv2.putText(
                display,
                f"Student: {name}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            cv2.putText(
                display,
                f"Photo {i+1}/5: {instruction}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 1
            )
            cv2.putText(
                display,
                "SPACE=Capture  Q=Skip",
                (display.shape[1]-220, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )

            cv2.imshow("Tenda Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                print(f"  Capturing...")
                embedding = extract_embedding(frame)

                if embedding is None:
                    print(f"  No face detected. Try again.")
                    continue

                embeddings.append(embedding)
                print(f"  Captured! ({len(embeddings)}/5)")
                break

            elif key == ord('q'):
                print(f"  Skipped")
                break

    if embeddings:
        print(f"\nSaving {len(embeddings)} embeddings...")
        result = save_all_embeddings(student_id, embeddings)
        if result:
            print(f"  {result.get('message', 'Saved!')}")
        else:
            print("  Save failed!")
    else:
        print("  No embeddings captured!")

    return len(embeddings)


def main():
    print("Enter Class ID:")
    class_id = input().strip()
    if not class_id:
        class_id = CLASS_ID
        print(f"Using default: {class_id}")
    
    print("Loading students...")
    students = get_students(class_id)

    if not students:
        print("No students found!")
        print("Make sure backend is running")
        return

    print(f"Found {len(students)} students:")
    for s in students:
        emb = len(s.get("embeddings", []))
        print(f"  {s['full_name']} - embeddings: {emb}")

    print("\nStarting Tenda camera...")
    cap = cv2.VideoCapture(
        CAMERA_SOURCE, cv2.CAP_FFMPEG
    )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera failed!")
        return

    print("Camera opened!")
    total_success = 0

    for student in students:
        emb = len(student.get("embeddings", []))
        if emb >= 5:
            print(f"  Skipping {student['full_name']} "
                  f"(already enrolled)")
            continue
        success = capture_photos_for_student(
            student, cap
        )
        total_success += success
        print(f"  {student['full_name']}: "
              f"{success} embeddings saved")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print(f"Enrollment complete!")
    print(f"Total embeddings saved: {total_success}")
    print("=" * 55)


if __name__ == "__main__":
    main()