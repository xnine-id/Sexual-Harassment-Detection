from src.services.sexual_harassment_tracker import SexualHarassmentTracker
from src.utils.config_loader import SnapshotConfig
import threading
import logging
import asyncio
from typing import Dict, List, Optional
from src.core.camera_processor import CameraProcessor
from src.core.sexual_harassment_detector import SexualHarassmentDetector
from src.services.mqtt_service import MQTTService
from src.services.frame_renderer import FrameRenderer
from src.database.entity.camera import Camera

logger = logging.getLogger("CAMERA_MANAGER")


class CameraManager:
    """Manages the lifecycle and streaming of multiple camera processors"""

    def __init__(
        self,
        snapshot_config: SnapshotConfig,
        sexual_harassment_detector: SexualHarassmentDetector,
        frame_renderer: FrameRenderer,
        mqtt_service: Optional[MQTTService] = None,
    ):
        self.snapshot_config = snapshot_config
        self.sexual_harassment_detector = sexual_harassment_detector
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.camera_processors: Dict[str, CameraProcessor] = {}

        # Setup Services
        self.mqtt_service = mqtt_service
        self.frame_renderer = frame_renderer

    async def _create_camera_processors(self):
        """
        Creates processor instances for each enabled camera defined in the database.
        """
        from src.database.session import get_sessionmaker
        from sqlalchemy.future import select

        session_factory = get_sessionmaker()
        async with session_factory() as session:
            result = await session.execute(select(Camera))
            cameras = result.scalars().all()

            for camera in cameras:
                self.add_camera_processor(camera)

    def add_camera_processor(self, camera: Camera):
        """Adds and starts a new camera processor."""
        if camera.name in self.camera_processors:
            logger.warning(f"Processor for camera '{camera.name}' already exists. Skipping.")
            return

        tracker = SexualHarassmentTracker(
            snapshot_config=self.snapshot_config,
            cam_name=camera.name,
            mqtt_service=self.mqtt_service,
        )

        processor = CameraProcessor(
            cam_config=camera,
            sexual_harassment_detector=self.sexual_harassment_detector,
            frame_renderer=self.frame_renderer,
            tracker=tracker,
            mqtt_service=self.mqtt_service,
        )
        self.camera_processors[camera.name] = processor
        
        # If manager is already running, start the processor thread
        if not self.stop_event.is_set():
            self._start_processor_thread(processor)

    def update_camera_processor(self, camera: Camera):
        """Updates an existing camera processor by restarting it with new config."""
        self.remove_camera_processor(camera.name)
        self.add_camera_processor(camera)

    def remove_camera_processor(self, camera_name: str):
        """Stops and removes a camera processor."""
        processor = self.camera_processors.pop(camera_name, None)
        if processor:
            processor.stop()
            # Find and join thread (optional, but good for cleanup)
            for i, thread in enumerate(self.threads):
                if thread.name == f"Thread-{camera_name}":
                    if thread.is_alive():
                        thread.join(timeout=2.0)
                    self.threads.pop(i)
                    break
            logger.info(f"Camera processor '{camera_name}' stopped and removed")

    def _start_processor_thread(self, processor: CameraProcessor):
        thread = threading.Thread(
            target=processor.run, name=f"Thread-{processor.cam_name}"
        )
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        logger.info(f"Camera processor '{processor.cam_name}' started")

    def _get_processor(self, name: str) -> Optional[CameraProcessor]:
        return self.camera_processors.get(name)

    async def stream_generator(self, camera_name: str):
        """Async generator for streaming frames from a specific camera"""
        processor = self._get_processor(camera_name)
        if not processor:
            logger.exception(f"Processor for camera '{camera_name}' not found")
            return

        while not self.stop_event.is_set():
            frame = processor.get_latest_frame()
            if frame:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

            # Limit to ~25fps to avoid excessive CPU/bandwidth usage
            await asyncio.sleep(0.04)

    async def start(self, blocking: bool = True):
        """Start all camera processors"""
        logger.info("Starting Camera Manager...")

        # Reset stop event in case of restart
        self.stop_event.clear()

        try:
            # Create processors from DB
            await self._create_camera_processors()

            logger.info(f"{len(self.camera_processors)} camera(s) initialized from database")

            if blocking:
                # Keep main thread alive and wait for stop_event
                while not self.stop_event.is_set():
                    self.stop_event.wait(1)

        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Stopping...")
            self.stop()
        except Exception as e:
            logger.exception(f"Unexpected error starting cameras: {e}")
            self.stop()
            if not blocking:
                raise e

    def stop(self):
        """Stop all processors and cleanup"""
        for processor in self.camera_processors.values():
            processor.stop()

        logger.info("Stopping Camera Manager...")

        # Publish stopped state for all cameras
        if self.mqtt_service:
            for cam_name in self.camera_processors.keys():
                self.mqtt_service.publish_state(cam_name, False)

        # Wait for all threads to finish with timeout
        try:
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not stop gracefully")

            self.stop_event.set()
        except KeyboardInterrupt:
            logger.warning("Force stopping (Ctrl+C during shutdown)...")
        finally:
            self.threads.clear()
            self.camera_processors.clear()

        logger.info("Camera Manager stopped")
