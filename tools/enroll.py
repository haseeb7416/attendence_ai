import cv2
import os
import requests
import numpy as np
from config import BASE_URL, FACE_MODEL

# Lazy load DeepFace
_DeepFace = None


def _get_deepface():
    global _DeepFace
    if _DeepFace is None:
        from deepface import DeepFace as df
        _DeepFace = df
    return _DeepFace


print("=" * 50)
print("STUDENT ENROLLMENT")
print("=" * 50)


def get_unprocessed_photos():
    try:
        r = requests.get(
            f"{BASE_URL}/students/no-embedding",
            headers={"ngrok-skip-browser-warning": "true"}
        )
        return r.json()
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_photo(photo_url, save_path):
    try:
        filename = photo_url.replace("uploads/", "")
        url      = f"{BASE_URL}/uploads/{filename}"
        headers  = {"ngrok-skip-browser-warning": "true"}
        r        = requests.get(
            url, headers=headers, timeout=10
        )

        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return False

        with open(save_path, "wb") as f:
            f.write(r.content)

        img = cv2.imread(save_path)
        if img is None:
            print(f"    Not valid image")
            os.remove(save_path)
            return False

        print(f"    Downloaded: {img.shape}")
        return True

    except Exception as e:
        print(f"    Download error: {e}")
        return False


def extract_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        proper_path = "temp_face_process.jpg"
        cv2.imwrite(proper_path, img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        df     = _get_deepface()
        result = df.represent(
            img_path=img_rgb,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend="opencv"
        )

        try:
            os.remove(proper_path)
        except:
            pass

        return result[0]["embedding"]

    except Exception as e:
        print(f"    Embedding error: {e}")
        return None


def save_embedding(student_id, photo_id, embedding):
    try:
        r = requests.put(
            f"{BASE_URL}/students/{student_id}"
            f"/photos/{photo_id}/embedding",
            json={"face_embedding": embedding},
            headers={"ngrok-skip-browser-warning": "true"}
        )
        return r.json()
    except Exception as e:
        print(f"    Save error: {e}")
        return None


def main():
    print("Fetching unprocessed photos...")
    photos = get_unprocessed_photos()

    if not photos:
        print("All students already have embeddings!")
        return

    print(f"Found {len(photos)} photos\n")

    success = 0
    failed  = 0

    for i, photo in enumerate(photos):
        photo_id   = photo["id"]
        student_id = photo["student_id"]
        photo_url  = photo["photo_url"]
        filename   = photo_url.replace("uploads/", "")

        print(f"[{i+1}/{len(photos)}] {filename}")

        save_path = f"temp_enroll_{i}"

        ok = download_photo(photo_url, save_path)
        if not ok:
            failed += 1
            continue

        print(f"    Extracting embedding...")
        embedding = extract_embedding(save_path)

        try:
            os.remove(save_path)
        except:
            pass

        if embedding is None:
            print(f"    No face detected")
            failed += 1
            continue

        print(f"    Saving to backend...")
        result = save_embedding(
            student_id, photo_id, embedding
        )

        if result:
            print(f"    Success!")
            success += 1
        else:
            print(f"    Failed")
            failed += 1

    print()
    print("=" * 50)
    print(f"Done! Success:{success} Failed:{failed}")
    print("=" * 50)


if __name__ == "__main__":
    main()