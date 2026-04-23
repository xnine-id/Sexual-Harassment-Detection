from fastapi import APIRouter
from src.core.camera_manager import CameraManager
from src.core.sexual_harassment_detector import SexualHarassmentDetector
from src.services.frame_renderer import FrameRenderer
from src.utils.config_loader import Config

# Import endpoint routers
from src.api.endpoints.camera import get_camera_router
from src.api.endpoints.media import get_media_router
from src.api.endpoints.system import get_system_router
from src.api.endpoints.prediction import get_prediction_router

def create_router(
    camera_manager: CameraManager,
    config: Config,
    sexual_detector: SexualHarassmentDetector,
    renderer: FrameRenderer,
):
    router = APIRouter()

    # Include Camera endpoints
    router.include_router(get_camera_router(camera_manager))

    # Include Media endpoints (Snapshots)
    router.include_router(get_media_router(config))

    # Include System endpoints (Health & Tokens)
    router.include_router(get_system_router())

    # Include Prediction endpoints
    router.include_router(get_prediction_router(config, sexual_detector, renderer))

    return router
