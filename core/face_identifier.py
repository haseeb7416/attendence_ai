import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import json
import numpy as np
from config import FACE_MODEL, FACE_THRESHOLD

FACE_RECOGNITION_INTERVAL = 10
VOTES_REQUIRED            = 2

# Lazy load DeepFace to avoid protobuf conflict
_DeepFace = None


def _get_deepface():
    global _DeepFace
    if _DeepFace is None:
        from deepface import DeepFace as df
        _DeepFace = df
    return _DeepFace


def crop_face_for_recognition(frame, bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Take upper 60% for face
    face_y2 = y1 + int(h * 0.60)
    margin  = int(w * 0.20)

    fx1 = max(0, x1 - margin)
    fy1 = max(0, y1 - margin)
    fx2 = min(frame.shape[1], x2 + margin)
    fy2 = min(frame.shape[0], face_y2)

    crop = frame[fy1:fy2, fx1:fx2]

    if crop.size == 0:
        return None

    # Upscale small faces
    if crop.shape[0] < 80 or crop.shape[1] < 80:
        crop = cv2.resize(
            crop, (160, 160),
            interpolation=cv2.INTER_LINEAR
        )

    return crop


def extract_embedding(image):
    try:
        df = _get_deepface()

        if isinstance(image, np.ndarray):
            img_rgb = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )
        else:
            img_rgb = image

        result = df.represent(
            img_path=img_rgb,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend="opencv"
        )
        return result[0]["embedding"]
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None


def cosine_distance(e1, e2):
    e1  = np.array(e1, dtype=np.float32)
    e2  = np.array(e2, dtype=np.float32)
    dot = np.dot(e1, e2)
    return 1.0 - dot / (
        np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9
    )


def find_best_match(face_embedding, students):
    distances = []

    for student in students:
        embeddings = student.get("embeddings", [])
        if not embeddings:
            continue

        best_dist = float("inf")

        for stored_emb in embeddings:
            try:
                if isinstance(stored_emb, str):
                    stored_emb = json.loads(stored_emb)

                dist = cosine_distance(
                    face_embedding, stored_emb
                )

                if dist < best_dist:
                    best_dist = dist

            except Exception:
                continue

        distances.append((best_dist, student))

    if not distances:
        return None, None, 0.0

    distances.sort(key=lambda x: x[0])

    best_dist    = distances[0][0]
    best_student = distances[0][1]

    # Reject if distance above threshold
    if best_dist > FACE_THRESHOLD:
        return None, None, 0.0

    # Margin check
    if len(distances) > 1:
        second_dist = distances[1][0]
        margin      = second_dist - best_dist
        if margin < 0.15:
            return None, None, 0.0

    raw_conf   = 1.0 - (best_dist / FACE_THRESHOLD)
    confidence = min(1.0, max(0.0, raw_conf * 2.5 + 0.3))

    return (
        best_student["id"],
        best_student["full_name"],
        round(confidence, 2)
    )


class FaceIdentifier:

    def __init__(self, students):
        self.students             = students
        self.identities           = {}
        self.frame_counters       = {}
        self.match_votes          = {}
        self.confirmed_embeddings = {}

        print("\n--- FaceIdentifier loaded students ---")
        for s in students:
            count = len(s.get("embeddings", []))
            print(f"  {s['full_name']}: {count} embeddings")
        print("--------------------------------------\n")

    def update(self, frame, tracked_persons):

        for person in tracked_persons:
            tid = person.track_id

            if tid not in self.frame_counters:
                self.frame_counters[tid] = 0
                self.match_votes[tid]    = {}

            self.frame_counters[tid] += 1

            # Skip if already confirmed
            if tid in self.identities:
                continue

            # Run every 10 frames
            if self.frame_counters[tid] % \
               FACE_RECOGNITION_INTERVAL != 0:
                continue

            # Crop face
            face_crop = crop_face_for_recognition(
                frame, person.bbox
            )
            if face_crop is None:
                continue

            # Extract embedding
            embedding = extract_embedding(face_crop)
            if embedding is None:
                continue

            # Check confirmed embeddings first
            best_confirmed_id   = None
            best_confirmed_dist = float("inf")

            for sid, conf_emb in \
                    self.confirmed_embeddings.items():
                dist = cosine_distance(embedding, conf_emb)
                if dist < best_confirmed_dist:
                    best_confirmed_dist = dist
                    best_confirmed_id   = sid

            if (best_confirmed_dist < FACE_THRESHOLD and
                    best_confirmed_id is not None):
                name = next(
                    (s["full_name"] for s in self.students
                     if s["id"] == best_confirmed_id),
                    "Unknown"
                )
                conf = max(
                    0.0,
                    1.0 - (best_confirmed_dist /
                           FACE_THRESHOLD)
                )
                conf = min(1.0, conf * 2.5 + 0.3)
                self.identities[tid] = (
                    best_confirmed_id,
                    name,
                    round(conf, 2)
                )
                print(f"Restored: ID {tid} = {name} ({conf:.0%})")
                continue

            # Normal identification flow
            student_id, name, confidence = find_best_match(
                embedding, self.students
            )

            if student_id is None:
                continue

            votes = self.match_votes[tid]
            if student_id not in votes:
                votes[student_id] = 0
            votes[student_id] += 1

            if votes[student_id] >= VOTES_REQUIRED:
                self.identities[tid] = (
                    student_id, name, confidence
                )
                self.confirmed_embeddings[student_id] = \
                    embedding
                print(f"Confirmed: ID {tid} = {name} ({confidence:.0%})")

        return self.identities

    def get_identity(self, track_id):
        return self.identities.get(
            track_id, (None, "Unknown", 0.0)
        )

    def reset_track(self, track_id):
        self.identities.pop(track_id, None)
        self.frame_counters.pop(track_id, None)
        self.match_votes.pop(track_id, None)