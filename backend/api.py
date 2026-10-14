import requests
from config import BASE_URL

HEADERS = {"ngrok-skip-browser-warning": "true"}


def get_students(class_id):
    try:
        r = requests.get(
            f"{BASE_URL}/students?class_id={class_id}",
            headers=HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"Error getting students: {e}")
        return []


def get_unprocessed_photos():
    try:
        r = requests.get(
            f"{BASE_URL}/students/no-embedding",
            headers=HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_photo(photo_url, save_path):
    try:
        filename = photo_url.replace("uploads/", "")
        r = requests.get(
            f"{BASE_URL}/uploads/{filename}",
            headers=HEADERS,
            timeout=15
        )
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Error downloading photo: {e}")
        return False


def save_embedding(student_id, photo_id, embedding):
    try:
        r = requests.put(
            f"{BASE_URL}/students/{student_id}"
            f"/photos/{photo_id}/embedding",
            json={"face_embedding": embedding},
            headers=HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"Error saving embedding: {e}")
        return None


def get_session_status(session_id):
    try:
        r = requests.get(
            f"{BASE_URL}/sessions/{session_id}",
            headers=HEADERS,
            timeout=15
        )
        return r.json()["status"]
    except Exception as e:
        print(f"Error getting session: {e}")
        return "unknown"


def get_session_info(session_id):
    try:
        r = requests.get(
            f"{BASE_URL}/sessions/{session_id}",
            headers=HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"Error getting session info: {e}")
        return {}


def get_active_session(class_id):
    try:
        r = requests.get(
            f"{BASE_URL}/sessions/active"
            f"?class_id={class_id}",
            headers=HEADERS,
            timeout=10
        )
        return r.json().get("session_id", None)
    except Exception as e:
        print(f"Error getting active session: {e}")
        return None


def submit_attendance(session_id, results):
    try:
        r = requests.post(
            f"{BASE_URL}/attendance/sessions/{session_id}/submit",
            json={"results": results},
            headers=HEADERS,
            timeout=30
        )
        return r.json()
    except Exception as e:
        print(f"Error submitting: {e}")
        return None