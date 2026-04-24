import logging
import uuid
import cv2
from typing import Optional, Dict, Any
import threading
import os
from datetime import datetime

from cv2.typing import MatLike

from src.services.sexual_harassment_tracker_int import SexualHarassmentTrackerInt
from src.services.mqtt_service import MQTTService
from src.utils.config_loader import SnapshotConfig

logger = logging.getLogger("SEXUAL_HARASSMENT_TRACKER")

HARASSMENT_TIME_THRESHOLD = 10  # seconds


class SexualHarassmentTracker(SexualHarassmentTrackerInt):
    def __init__(
        self, snapshot_config: SnapshotConfig, cam_name: str, mqtt_service: MQTTService
    ):
        self.snapshot_enabled = snapshot_config.enabled
        self.output_dir = snapshot_config.output_dir
        self.cam_name = cam_name
        self.mqtt_service = mqtt_service

        self.event_id: Optional[str] = None
        self.last_harassment_time: Optional[datetime] = None

    def update(self, frame: MatLike, result: Dict[str, Any]):
        """Update detection"""
        now = datetime.now()
        is_new_event = (
            self.last_harassment_time is None
            or self.event_id is None
            or (now - self.last_harassment_time).seconds > HARASSMENT_TIME_THRESHOLD
        )

        if result:
            def mqtt_task(frame: MatLike):
                if result["class"] == 1:
                    snapshot = None
                    self.last_harassment_time = now

                    if is_new_event:
                        self.event_id = str(uuid.uuid4())
                        if self.snapshot_enabled and frame is not None:
                            snapshot = self._save_snapshot(frame)

                    if self.mqtt_service:
                        self.mqtt_service.publish_event(
                            event_id=self.event_id,
                            cam_name=self.cam_name,
                            confidence=result["score"] * 100,
                            snapshot=snapshot,
                        )

                else:
                    if is_new_event:
                        self.event_id = str(uuid.uuid4())

                        if self.mqtt_service:
                            self.mqtt_service.publish_event(
                                event_id=self.event_id,
                                cam_name=self.cam_name,
                                confidence=result["score"] * 100,
                                event_type="no_harassment",
                            )

            threading.Thread(target=mqtt_task, args=(frame.copy(),), daemon=True).start()

    def reset(self):
        self.event_id = None
        self.last_harassment_time = None

    def _save_snapshot(self, frame: MatLike):
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:19] # include ms

        """Save snapshot in a separate thread to avoid blocking"""

        try:
            final_output_dir = os.path.join(self.output_dir, today)

            os.makedirs(final_output_dir, exist_ok=True)

            final_output = os.path.join(
                final_output_dir, f"{self.cam_name}_{timestamp}.jpg"
            )
            cv2.imwrite(final_output, frame)
        except Exception as e:
            logger.info(f"[{self.cam_name}] Failed to save snapshot: {e}")
            return None

        return f"/snapshots/{today}/{self.cam_name}_{timestamp}.jpg"
