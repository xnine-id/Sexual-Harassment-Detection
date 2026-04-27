from src.core.camera_manager import CameraManager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.database.entity.camera import Camera
from src.api.schemas import AddCameraRequest, UpdateCameraRequest
from typing import List, Optional

class CameraService:
    def __init__(self, session: AsyncSession, camera_manager: CameraManager):
        self.session = session
        self.camera_manager = camera_manager

    async def get_cameras(self) -> List[Camera]:
        result = await self.session.execute(select(Camera))
        return result.scalars().all()

    async def get_camera(self, camera_name: str) -> Optional[Camera]:
        result = await self.session.execute(select(Camera).where(Camera.name == camera_name))
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

        # Add new camera to manager
        self.camera_manager.add_camera_processor(new_camera)

        return new_camera

    async def update_camera(self, camera_name: str, data: UpdateCameraRequest) -> Optional[Camera]:
        camera = await self.get_camera(camera_name)
        if not camera:
            return None

        old_name = camera.name
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(camera, key, value)

        await self.session.commit()
        await self.session.refresh(camera)

        # Update in camera manager
        if old_name != camera.name:
            self.camera_manager.remove_camera_processor(old_name)
            self.camera_manager.add_camera_processor(camera)
        else:
            self.camera_manager.update_camera_processor(camera)

        return camera

    async def delete_camera(self, camera_name: str) -> bool:
        camera = await self.get_camera(camera_name)
        if not camera:
            return False

        cam_name = camera.name
        await self.session.delete(camera)
        await self.session.commit()

        # Remove from camera manager
        self.camera_manager.remove_camera_processor(cam_name)

        return True

    async def sync_cameras(self, data: List[AddCameraRequest]) -> List[Camera]:
        existing_cameras = await self.get_cameras()
        existing_map = {cam.name: cam for cam in existing_cameras}
        new_names = {item.name for item in data}

        # 1. Delete cameras not in the new list
        for name, camera in existing_map.items():
            if name not in new_names:
                await self.session.delete(camera)
                self.camera_manager.remove_camera_processor(name)

        # 2. Add or Update cameras
        synced_cameras = []
        for item in data:
            if item.name in existing_map:
                # Update
                camera = existing_map[item.name]
                camera.url = item.url
                camera.detect_fps = item.detect_fps
                camera.is_enabled = item.is_enabled
                camera.snapshot_enabled = item.snapshot_enabled
                camera.mqtt_enabled = item.mqtt_enabled
                self.camera_manager.update_camera_processor(camera)
            else:
                # Add
                camera = Camera(
                    name=item.name,
                    url=item.url,
                    detect_fps=item.detect_fps,
                    is_enabled=item.is_enabled,
                    snapshot_enabled=item.snapshot_enabled,
                    mqtt_enabled=item.mqtt_enabled
                )
                self.session.add(camera)
                self.camera_manager.add_camera_processor(camera)
            
            synced_cameras.append(camera)

        await self.session.commit()
        
        # Refresh all synced cameras to get their IDs and updated state
        for cam in synced_cameras:
            await self.session.refresh(cam)
        
        return synced_cameras
