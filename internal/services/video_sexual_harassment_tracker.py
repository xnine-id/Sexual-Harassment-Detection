import logging
import json
from typing import Dict, Any
from cv2.typing import MatLike

from internal.services.sexual_harassment_tracker_int import SexualHarassmentTrackerInt

logger = logging.getLogger("VIDEO_SEXUAL_HARASSMENT_TRACKER")


class VideoSexualHarassmentTracker(SexualHarassmentTrackerInt):
    def __init__(self):
        self.scores: list[int] = []
        self.all_predictions: list[Dict[str, Any]] = []

    def update(self, frame: MatLike, result: Dict[str, Any]):
        """Update detection"""
        if result:
            self.all_predictions.append({
                "class": int(result["class"]),
                "score": float(result["score"]),
            })

        if result and result["class"] == 1:
            self.scores.append(result["score"])

    def get_scores(self) -> list[int]:
        return self.scores

    def get_avg_scores(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0

    def get_highest_score(self) -> float:
        return max(self.scores) if self.scores else 0

    def get_lowest_score(self) -> float:
        return min(self.scores) if self.scores else 0
    
    def get_frequency(self):
        total_frames = len(self.all_predictions)
        fight_frames = len(self.scores)
        return fight_frames / total_frames

    def reset(self):
        self.scores = []
        self.all_predictions = []

    def save_all_predictions(self, json_output_path: str):
        """Save all predictions to a JSON file."""

        logger.info(f"save all predictions ({len(self.all_predictions)}) to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(self.all_predictions, f, indent=2)    
