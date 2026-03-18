import logging
import uuid
import cv2
from typing import Optional, Dict, Any
import threading
import os
from datetime import datetime

from cv2.typing import MatLike

from internal.services.sexual_harassment_tracker_int import SexualHarassmentTrackerInt
from internal.services.mqtt_service import MQTTService

logger = logging.getLogger("SEXUAL_HARASSMENT_TRACKER")


class SexualHarassmentTracker(SexualHarassmentTrackerInt):
    def __init__(
        self, snapshot_config: Dict[str, Any], cam_name: str, mqtt_service: MQTTService
    ):
        self.snapshot_enabled = snapshot_config["enabled"]
        self.output_dir = snapshot_config["output_dir"]
        self.cam_name = cam_name
        self.mqtt_service = mqtt_service

        self.threshold = 5

        self.current_score: Optional[int] = None
        self.update_count = 0
        self.event_id: Optional[str] = None

    def update(self, frame: MatLike, result: Dict[str, Any]):
        """Update detection"""
        if result and result["class"] == 1:
            self.update_count = 0
            prev_score = self.current_score
            self.current_score = result["score"]

            snapshot = None
            if prev_score == None or self.event_id == None:
                self.event_id = str(uuid.uuid4())
                if self.snapshot_enabled and frame is not None:
                    snapshot = self._save_snapshot(frame)

            if self.mqtt_service and self.current_score != prev_score:
                self.mqtt_service.publish_event(
                    event_id=self.event_id,
                    cam_name=self.cam_name,
                    confidence=result["score"] * 100,
                    snapshot=snapshot,
                )

        else:
            self.update_count += 1
            if self.update_count % self.threshold == 0:
                prev_score = self.current_score
                self.current_score = None

                if prev_score != None:
                    self.event_id = str(uuid.uuid4())
                    self.mqtt_service.publish_event(
                        event_id=self.event_id,
                        cam_name=self.cam_name,
                        confidence=result["score"] * 100,
                        event_type="no_harassment",
                    )

    def reset(self):
        self.current_score = None
        self.event_id = None
        self.update_count = 0

    def get_current_detection(self):
        """Get current detection for rendering"""
        return self.current_detection

    def _save_snapshot(self, frame: MatLike):
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        """Save snapshot in a separate thread to avoid blocking"""

        def save_task():
            try:
                final_output_dir = os.path.join(self.output_dir, today)

                os.makedirs(final_output_dir, exist_ok=True)

                final_output = os.path.join(
                    final_output_dir, f"{self.cam_name}_{timestamp}.jpg"
                )
                cv2.imwrite(final_output, frame)
            except Exception as e:
                logger.Info(f"[{self.cam_name}] Failed to save snapshot: {e}")

        # Run in background
        threading.Thread(target=save_task, daemon=True).start()

        # Return path immediately (predicted path)
        return f"/snapshots/{today}/{self.cam_name}_{timestamp}.jpg"