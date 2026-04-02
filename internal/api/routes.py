import logging
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from internal.core.camera_manager import CameraManager
from internal.utils.config_loader import Config

logger = logging.getLogger("API_ROUTES")


def create_router(camera_manager: CameraManager, config: Config):
    router = APIRouter()

    snapshot_dir = config.snapshot.output_dir

    @router.get(
        "/snapshots/{date_str}/{filename}",
        summary="Get snapshot image file",
        tags=["Snapshot"],
    )
    async def get_snapshot(date_str: str, filename: str):
        """
        Get snapshot image by date and filename.
        """
        if not snapshot_dir:
            raise HTTPException(
                status_code=500, detail="Snapshot directory not configured"
            )

        file_path = os.path.join(snapshot_dir, date_str, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return FileResponse(file_path)

    @router.get("/video_feed/{camera_name}", tags=["Streaming"])
    async def video_feed(camera_name: str):
        """
        Get video feed from a specific camera.
        """
        return StreamingResponse(
            camera_manager.stream_generator(camera_name),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return router
