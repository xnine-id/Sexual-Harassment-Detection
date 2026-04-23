from src.api.middleware.auth import verify_token
from src.core.camera_manager import CameraManager
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_db
from src.services.camera_service import CameraService
from src.api.schemas import AddCameraRequest, UpdateCameraRequest, CameraResponse, GenericResponse
from typing import List


def get_camera_router(camera_manager: CameraManager):
    router = APIRouter(prefix="/cameras", tags=["Camera"], dependencies=[Depends(verify_token)])

    async def get_camera_service(db: AsyncSession = Depends(get_db)):
        return CameraService(db, camera_manager)

    @router.get("", response_model=List[CameraResponse])
    async def list_cameras(service: CameraService = Depends(get_camera_service)):
        return await service.get_cameras()

    @router.get("/{camera_id}", response_model=CameraResponse)
    async def get_camera(camera_id: int, service: CameraService = Depends(get_camera_service)):
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @router.post("", response_model=CameraResponse)
    async def add_camera(request: AddCameraRequest, service: CameraService = Depends(get_camera_service)):
        camera = await service.create_camera(request)
        # camera_manager.restart()
        return camera

    @router.put("/{camera_id}", response_model=CameraResponse)
    async def update_camera(camera_id: int, request: UpdateCameraRequest, service: CameraService = Depends(get_camera_service)):
        camera = await service.update_camera(camera_id, request)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @router.delete("/{camera_id}", response_model=GenericResponse)
    async def delete_camera(camera_id: int, service: CameraService = Depends(get_camera_service)):
        success = await service.delete_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail="Camera not found")
        return GenericResponse(status="success", message="Camera deleted successfully")

    return router