from abc import ABC, abstractmethod
from typing import Any, Dict
from cv2.typing import MatLike


class SexualHarassmentTrackerInt(ABC):
    @abstractmethod
    def update(self, frame: MatLike, result: Dict[str, Any]):
        pass
