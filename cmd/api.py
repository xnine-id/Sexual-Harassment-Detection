from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Add the project root to PYTHONPATH to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal.utils.config_loader import load_config
from internal.services.mqtt_service import MQTTService
from internal.core.camera_manager import CameraManager
from internal.core.sexual_harassment_detector import SexualHarassmentDetector
from internal.api.routes import create_router
from internal.utils.logging_utils import setup_logging

logger = logging.getLogger("API")


def create_app():
    setup_logging()
    # Load configuration
    config = load_config("configs/config.yml")

    # Initialize face recognition
    sexual_harassment_detector = SexualHarassmentDetector(config)

    # Initialize services
    mqtt_service = MQTTService(config)

    # Initialize Camera Manager (which manages FaceRecognition)
    camera_manager = CameraManager(
        config,
        sexual_harassment_detector=sexual_harassment_detector,
        mqtt_service=mqtt_service,
    )

    # Lifespan handler
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Startup
            camera_manager.start(blocking=False)

            yield
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Unknown error in lifespan: {e}")
        finally:
            camera_manager.stop()
            if mqtt_service:
                mqtt_service.disconnect()

    # Initialize FastAPI app
    app = FastAPI(title="Sexual Harassment Detection API", lifespan=lifespan)

    # Add CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    api_router = create_router(camera_manager, config)
    app.include_router(api_router, prefix="/api")

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "message": "Sexual Harassment Detection API is running"}

    return app


if __name__ == "__main__":
    try:
        app = create_app()
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "5000"))
        uvicorn.run(
            app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info")
        )
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        sys.exit(1)
