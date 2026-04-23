from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.database.entity.camera import Camera
from src.api.schemas import AddCameraRequest, UpdateCameraRequest
from typing import List, Optional

class CameraService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_cameras(self) -> List[Camera]:
        result = await self.session.execute(select(Camera))
        return result.scalars().all()

    async def get_camera(self, camera_id: int) -> Optional[Camera]:
        result = await self.session.execute(select(Camera).where(Camera.id == camera_id))
        return result.scalar_one_or_none()

    async def create_camera(self, data: AddCameraRequest) -> Camera:
        new_camera = Camera(
            name=data.name,
            url=data.url,
            detect_fps=data.detect_fps,
            is_enabled=data.is_enabled,
            snapshot_enabled=data.snapshot_enabled,
            mqtt_enabled=data.mqtt_enabled
        )
        self.session.add(new_camera)
        await self.session.commit()
        await self.session.refresh(new_camera)
        return new_camera

    async def update_camera(self, camera_id: int, data: UpdateCameraRequest) -> Optional[Camera]:
        camera = await self.get_camera(camera_id)
        if not camera:
            return None
        
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(camera, key, value)
        
        await self.session.commit()
        await self.session.refresh(camera)
        return camera

    async def delete_camera(self, camera_id: int) -> bool:
        camera = await self.get_camera(camera_id)
        if not camera:
            return False
        
        await self.session.delete(camera)
        await self.session.commit()
        return True
