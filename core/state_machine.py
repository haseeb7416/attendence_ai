import time
from dataclasses import dataclass, field
from typing import Optional, Dict
from config import STATES


@dataclass
class StudentAttendance:
    student_id:       str
    student_name:     str
    track_id:         int
    first_seen:       float = field(
        default_factory=time.time
    )
    last_seen:        float = field(
        default_factory=time.time
    )

    engaged_count:    int = 0
    distracted_count: int = 0
    sleeping_count:   int = 0
    talking_count:    int = 0

    confidence_sum:   float = 0.0
    confidence_count: int   = 0

    final_status:     str   = "ABSENT"
    final_engagement: str   = "ENGAGED"
    avg_confidence:   float = 0.0

    def update_state(self, state: str, confidence: float):
        self.last_seen = time.time()
        confidence     = float(confidence)

        if state == "ENGAGED":
            self.engaged_count    += 1
        elif state == "DISTRACTED":
            self.distracted_count += 1
        elif state == "SLEEPING":
            self.sleeping_count   += 1
        elif state == "TALKING":
            self.talking_count    += 1

        self.confidence_sum   += confidence
        self.confidence_count += 1

    def compute_final(self):
        total = (
            self.engaged_count    +
            self.distracted_count +
            self.sleeping_count   +
            self.talking_count
        )

        if total == 0:
            self.final_status     = "ABSENT"
            self.final_engagement = "ENGAGED"
            self.avg_confidence   = 0.0
            return

        self.final_status = "PRESENT"

        if self.confidence_count > 0:
            self.avg_confidence = round(
                self.confidence_sum /
                self.confidence_count, 2
            )

        engaged_pct  = self.engaged_count    / total
        distract_pct = self.distracted_count / total
        sleep_pct    = self.sleeping_count   / total
        talking_pct  = self.talking_count    / total

        if sleep_pct >= 0.30:
            self.final_engagement = "SLEEPING"
        elif engaged_pct >= 0.50:
            self.final_engagement = "ENGAGED"
        elif talking_pct >= 0.30:
            self.final_engagement = "TALKING"
        else:
            self.final_engagement = "DISTRACTED"

    def to_result(self) -> dict:
        self.compute_final()
        return {
            "student_id":   self.student_id,
            "final_status": self.final_status,
            "confidence":   self.avg_confidence,
            "engagement_counts": {
                "ENGAGED":    self.engaged_count,
                "DISTRACTED": self.distracted_count,
                "SLEEPING":   self.sleeping_count,
                "TALKING":    self.talking_count,
            }
        }


class StateMachine:

    def __init__(self, students: list):
        self.students          = students
        self.records:          Dict[str, StudentAttendance] = {}
        self.track_to_student: Dict[int, str] = {}

    def update(
        self,
        track_id:     int,
        student_id:   Optional[str],
        student_name: str,
        pose_state:   str,
        is_sleeping:  bool,
        confidence:   float
    ):
        if student_id is None:
            return

        self.track_to_student[track_id] = student_id

        if student_id not in self.records:
            self.records[student_id] = StudentAttendance(
                student_id=student_id,
                student_name=student_name,
                track_id=track_id
            )

        if is_sleeping:
            final_state = "SLEEPING"
        elif pose_state in STATES:
            final_state = pose_state
        else:
            final_state = "ENGAGED"

        self.records[student_id].update_state(
            final_state, confidence
        )

    def get_all_results(self) -> list:
        results = []

        for student_id, record in self.records.items():
            results.append(record.to_result())

        present_ids = set(self.records.keys())
        for student in self.students:
            sid = student["id"]
            if sid not in present_ids:
                results.append({
                    "student_id":   sid,
                    "final_status": "ABSENT",
                    "confidence":   0.0,
                    "engagement_counts": {
                        "ENGAGED":    0,
                        "DISTRACTED": 0,
                        "SLEEPING":   0,
                        "TALKING":    0,
                    }
                })

        return results

    def get_live_status(self) -> dict:
        status = {}
        for sid, record in self.records.items():
            total = (
                record.engaged_count    +
                record.distracted_count +
                record.sleeping_count   +
                record.talking_count
            )
            status[sid] = {
                "name":       record.student_name,
                "track_id":   record.track_id,
                "total_seen": total,
                "last_seen":  record.last_seen
            }
        return status

    def reset(self):
        self.records          = {}
        self.track_to_student = {}