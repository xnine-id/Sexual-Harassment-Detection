from fastapi import Depends
from src.api.middleware.auth import verify_token
from src.utils.config_loader import Config
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

def get_media_router(config: Config):
    router = APIRouter(tags=["Media"])

    # Get snapshot directory from config
    snapshot_dir = config.snapshot.output_dir

    @router.get(
        "/snapshots/{date_str}/{filename}",
        summary="Get snapshot image file",
        dependencies=[Depends(verify_token)]
    )
    async def get_snapshot(date_str: str, filename: str):
        """
        Get snapshot image by date and filename.
        """
        if not snapshot_dir:
            raise HTTPException(
                status_code=500, detail="Snapshot directory not configured"
            )

        # Security: Prevent directory traversal by ensuring the resolved path is within snapshot_dir
        base_dir = os.path.abspath(snapshot_dir)
        requested_path = os.path.abspath(os.path.join(base_dir, date_str, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return FileResponse(requested_path, filename=filename)

    return router
