from typing import Any, Dict
import cv2
from cv2.typing import MatLike


class FrameRenderer:
    """Render bounding boxes and labels on frames"""

    def __init__(self, box_color=(0, 255, 0), text_color=(0, 0, 0)):
        self.box_color = box_color
        self.text_color = text_color

    def render(self, frame: MatLike, detections: Dict[str, Any]):
        """Draw all detections on frame"""

        prob_text = f"{detections['label']} ({detections['score']:.2f})"
        color = (0, 0, 255) if detections["class"] == 1 else (0, 255, 0)

        font_scale = 1

        cv2.putText(
            frame,
            prob_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )

        return frame
