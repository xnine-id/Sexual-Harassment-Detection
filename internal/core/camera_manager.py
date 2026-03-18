import threading
import logging
import asyncio
from typing import Dict, List, Any, Optional
from internal.core.camera_processor import CameraProcessor
from internal.core.sexual_harassment_detector import SexualHarassmentDetector
from internal.services.mqtt_service import MQTTService

logger = logging.getLogger("CAMERA_MANAGER")


class CameraManager:
    """Manages the lifecycle and streaming of multiple camera processors"""

    def __init__(
        self,
        config: Dict[str, Any],
        sexual_harassment_detector: SexualHarassmentDetector,
        mqtt_service: Optional[MQTTService] = None,
    ):
        self.config = config
        self.sexual_harassment_detector = sexual_harassment_detector
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.camera_processors: Dict[str, CameraProcessor] = {}

        # Setup MQTT Service
        self.mqtt_service = mqtt_service

    def _create_camera_processors(self):
        """Create processor instances for each enabled camera"""

        for cam_config in self.config["cameras"]:
            processor = CameraProcessor(
                cam_config=cam_config,
                snapshot_config=self.config["snapshot"],
                mqtt_service=self.mqtt_service,
                sexual_harassment_detector=self.sexual_harassment_detector,
                stop_event=self.stop_event,
            )
            self.camera_processors[processor.cam_name] = processor

    def _get_processor(self, name: str) -> Optional[CameraProcessor]:
        return self.camera_processors.get(name)

    async def stream_generator(self, camera_name: str):
        """Async generator for streaming frames from a specific camera"""
        processor = self._get_processor(camera_name)
        if not processor:
            logger.error(f"Processor for camera '{camera_name}' not found")
            return

        while not self.stop_event.is_set():
            frame = processor.get_latest_frame()
            if frame:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

            # Limit to ~25fps to avoid excessive CPU/bandwidth usage
            await asyncio.sleep(0.04)

    def start(self, blocking: bool = True):
        """Start all camera processors"""
        logger.info("Starting Camera Manager...")

        # Reset stop event in case of restart
        self.stop_event.clear()

        try:
            # Create processors
            self._create_camera_processors()

            # Start each processor in separate thread
            for processor in self.camera_processors.values():
                thread = threading.Thread(
                    target=processor.run, name=f"Thread-{processor.cam_name}"
                )
                thread.daemon = True
                thread.start()
                self.threads.append(thread)

            logger.info(f"{len(self.threads)} camera(s) started")

            if blocking:
                # Keep main thread alive and wait for stop_event
                while not self.stop_event.is_set():
                    self.stop_event.wait(1)

        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Stopping...")
            self.stop()
        except Exception as e:
            logger.error(f"Unexpected error starting cameras: {e}")
            self.stop()
            if not blocking:
                raise e

    def stop(self):
        """Stop all processors and cleanup"""
        self.stop_event.set()
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
        except KeyboardInterrupt:
            logger.warning("Force stopping (Ctrl+C during shutdown)...")
        finally:
            self.threads.clear()
            self.camera_processors.clear()

        if self.mqtt_service:
            self.mqtt_service.disconnect()

        logger.info("Camera Manager stopped")
