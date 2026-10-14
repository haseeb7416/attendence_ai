import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
import time
import requests
import threading
from core.tracker import YOLOByteTracker
from core.pose_analyzer import PersonPoseState
from core.face_identifier import FaceIdentifier
from core.state_machine import StateMachine
from backend.api import (
    get_students,
    get_session_status,
    get_session_info,
    get_active_session,
    submit_attendance
)
from config import (
    CLASS_ID, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_IMGSZ, CAMERA_SOURCE,
    SESSION_POLL_SEC, BASE_URL
)

STATE_COLORS = {
    "ENGAGED":     (0, 255, 0),
    "DISTRACTED":  (0, 165, 255),
    "SLEEPING":    (0, 0, 255),
    "TALKING":     (255, 0, 255),
    "CALIBRATING": (255, 255, 0),
    "UNKNOWN":     (128, 128, 128),
}


def get_active_session(class_id):
    try:
        r = requests.get(
            f"{BASE_URL}/sessions/active"
            f"?class_id={class_id}",
            headers={
                "ngrok-skip-browser-warning": "true"
            },
            timeout=10
        )
        data = r.json()
        return data.get("session_id", None)
    except:
        return None


class SessionPoller:
    def __init__(self, session_id):
        self.session_id = session_id
        self.status     = "active"
        self.running    = True
        self.thread     = threading.Thread(
            target=self._poll,
            daemon=True
        )
        self.thread.start()

    def _poll(self):
        while self.running:
            try:
                status      = get_session_status(
                    self.session_id
                )
                self.status = status
                print(f"Session status: {status}")
            except:
                pass
            time.sleep(SESSION_POLL_SEC)

    def stop(self):
        self.running = False


def main(session_id: str):
    print("=" * 55)
    print("AI ATTENDANCE SYSTEM")
    print("=" * 55)

    # ── Load students ───────────────────────────────────────────
    print("Loading students from backend...")
    students = get_students(CLASS_ID)
    if not students:
        print("No students found! Check backend.")
        return

    print(f"Loaded {len(students)} students")
    for s in students:
        emb = len(s.get("embeddings", []))
        print(f"  {s['full_name']}: {emb} embeddings")

    # ── Initialize components ───────────────────────────────────
    print("\nInitializing AI components...")
    tracker       = YOLOByteTracker()
    face_id       = FaceIdentifier(students)
    state_machine = StateMachine(students)

    pose_states = {}

    # ── Camera ──────────────────────────────────────────────────
    print(f"Opening camera: {CAMERA_SOURCE}")
    if isinstance(CAMERA_SOURCE, int):
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    else:
        cap = cv2.VideoCapture(
            CAMERA_SOURCE, cv2.CAP_FFMPEG
        )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Camera failed!")
        return

    print("System ready!")
    print(f"Session ID: {session_id}")
    print("Press Q to quit manually")
    print("=" * 55)

    frame_idx      = 0
    session_ended  = False
    poller         = SessionPoller(session_id)

    # ── FPS tracking ─────────────────────────────────────────────
    fps_start      = time.time()
    fps_counter    = 0
    fps            = 0
    dropped_frames = 0
    total_frames   = 0

    while not session_ended:
        ok, frame = cap.read()
        total_frames += 1

        if not ok:
            dropped_frames += 1
            continue

        frame_idx += 1
        now = time.time()

        # ── FPS calculation ──────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps         = fps_counter
            fps_counter = 0
            fps_start   = time.time()
            print(f"FPS: {fps} | "
                  f"Dropped: {dropped_frames} | "
                  f"Total: {total_frames}")

        # ── Check session status from background thread ──────────
        if poller.status == "ended":
            print("Session ended! Submitting...")
            poller.stop()
            session_ended = True
            break

        # ── Track persons ────────────────────────────────────────
        persons = tracker.update(frame)

        # ── Face identification ──────────────────────────────────
        face_id.update(frame, persons)

        # ── Process each person ──────────────────────────────────
        for person in persons:
            tid  = person.track_id
            bbox = person.bbox
            kpts = person.kpts

            if tid not in pose_states:
                pose_states[tid] = PersonPoseState()

            pose_label  = pose_states[tid].update(
                kpts, bbox, now
            )
            is_sleeping = (pose_label == "SLEEPING")
            final_state = pose_label

            student_id, name, conf = \
                face_id.get_identity(tid)

            state_machine.update(
                track_id=tid,
                student_id=student_id,
                student_name=name,
                pose_state=final_state,
                is_sleeping=is_sleeping,
                confidence=conf
            )

            x1, y1, x2, y2 = bbox
            color = STATE_COLORS.get(
                final_state, (128, 128, 128)
            )
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )

            if name != "Unknown":
                label = f"{name} | {final_state}"
            else:
                votes  = face_id.match_votes.get(tid, {})
                best_v = max(votes.values()) \
                         if votes else 0
                label  = f"ID:{tid} {best_v}/4 | " \
                         f"{final_state}"

            cv2.putText(
                frame, label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2
            )

        # ── Live stats ───────────────────────────────────────────
        live = state_machine.get_live_status()
        cv2.rectangle(
            frame, (0, 0),
            (frame.shape[1], 22),
            (0, 0, 0), -1
        )
        cv2.putText(
            frame,
            f"Session:{session_id[:8]}.. "
            f"Tracked:{len(persons)} "
            f"ID:{len(live)} "
            f"FPS:{fps} "
            f"Drop:{dropped_frames}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1
        )

        cv2.imshow("AI Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Manual quit")
            poller.stop()
            session_ended = True

    # ── Submit attendance ────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── Final stats ──────────────────────────────────────────────
    print(f"\nSession Stats:")
    print(f"  Total frames:   {total_frames}")
    print(f"  Dropped frames: {dropped_frames}")
    if total_frames > 0:
        print(f"  Drop rate:      "
              f"{(dropped_frames/total_frames*100):.1f}%")

    print("\nComputing final results...")
    results = state_machine.get_all_results()

    print(f"Submitting {len(results)} records...")
    for r in results:
        name = next(
            (s["full_name"] for s in students
             if s["id"] == r["student_id"]),
            "Unknown"
        )
        print(f"  {name}: {r['final_status']} "
              f"({r['confidence']:.0%})")

    response = submit_attendance(session_id, results)
    print(f"Backend response: {response}")
    print("\nDone!")


if __name__ == "__main__":
    print("=" * 55)
    print("AI ATTENDANCE SYSTEM")
    print("Waiting for active session...")
    print("=" * 55)

    while True:
        session_id = get_active_session(CLASS_ID)
        if session_id:
            print(f"Session found: {session_id}")
            main(session_id)
            print("\nWaiting for next session...")
        else:
            print("No active session. Waiting 5s...")
            time.sleep(5)