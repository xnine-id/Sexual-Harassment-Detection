import cv2
from cv2.typing import MatLike
import threading
import time
import logging
import queue
import subprocess
from typing import Dict, Any, Optional
from threading import Event, Lock
from internal.services.frame_renderer import FrameRenderer
from internal.services.mqtt_service import MQTTService
from internal.core.sexual_harassment_detector import SexualHarassmentDetector
from internal.services.sexual_harassment_tracker import SexualHarassmentTracker
from internal.utils.config_loader import CameraConfig, SnapshotConfig

logger = logging.getLogger("CAM_PROCESSOR")


class CameraProcessor:
    """Process single camera stream with sexual harassment detection"""

    def __init__(
        self,
        cam_config: CameraConfig,
        snapshot_config: SnapshotConfig,
        sexual_harassment_detector: SexualHarassmentDetector,
        frame_renderer: FrameRenderer,
        mqtt_service: Optional[MQTTService] = None,
        stop_event: Optional[Event] = None,
        output_path: Optional[str] = None,
    ):
        # Configuration
        self.cam_name = cam_config.name
        self.url = cam_config.url
        self.detect_fps = cam_config.detect_fps
        self.is_running = cam_config.enabled
        self.width = 0
        self.height = 0

        self.stop_event = stop_event or Event()
        self.is_live_stream = False
        self.output_path = output_path
        self.out: Optional[cv2.VideoWriter] = None
        self.ffmpeg_proc: Optional[subprocess.Popen] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=128)

        # 1. Capture State
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_captured_frame: Optional[MatLike] = None
        self.capture_lock = Lock()
        self.capture_ret = False
        self.new_captured_frame_event = Event()
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0

        # 2. Detection State
        self.sexual_harassment_detector = sexual_harassment_detector
        self.latest_raw_frame: Optional[MatLike] = None
        self.raw_frame_lock = Lock()
        self.current_detection: Optional[Dict[str, Any]] = None
        self.detections_lock = Lock()
        self.new_frame_event = Event()

        # 3. Processing & Lazy Encoding State
        self.frame_to_render: Optional[MatLike] = None
        self.render_lock = Lock()

        # 4. Services
        self.mqtt_service: Optional[MQTTService] = mqtt_service
        self.frame_renderer = frame_renderer
        self.sexual_harassment_tracker = SexualHarassmentTracker(
            snapshot_config=snapshot_config,
            cam_name=self.cam_name,
            mqtt_service=self.mqtt_service,
        )

        # Register for MQTT commands
        if self.mqtt_service:
            self.mqtt_service.register_camera(
                self.cam_name, self.is_running, self._on_mqtt_command
            )

    def _on_mqtt_command(self, payload: Dict[str, Any]):
        """Handle incoming MQTT commands for this camera"""
        run_status: Optional[bool] = payload.get("run")
        if run_status is not None:
            self.is_running = run_status
            if self.mqtt_service:
                self.mqtt_service.publish_state(self.cam_name, self.is_running)
            status_str = "ENABLED" if run_status else "DISABLED"
            if not self.is_running:
                self.sexual_harassment_tracker.reset()
            logger.info(f"[{self.cam_name}] Status changed to {status_str} via MQTT")

    def _initialize_capture(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(self.url)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        if frame_count <= 0:
            self.is_live_stream = True

        if self.output_path:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                self.output_path,
            ]
            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug(
                f"[{self.cam_name}] FFmpeg writer initialized ({self.width}x{self.height} @ {fps}fps): {self.output_path}"
            )

        logger.info(f"[{self.cam_name}] Camera initialized (FPS: {fps}, Frame Count: {frame_count})")

    def _handle_reconnection(self):
        """Handle camera reconnection logic with exponential backoff"""
        logger.warning(
            f"[{self.cam_name}] Failed to reopen stream, retrying in {self.reconnect_delay:.1f}s"
        )
        self.stop_event.wait(self.reconnect_delay)
        self.reconnect_delay = min(
            self.reconnect_delay * 2, self.max_reconnect_delay
        )

    def _capture_loop(self):
        """Dedicated thread for high-frequency frame capture"""
        logger.debug(f"[{self.cam_name}] Capture loop started")
        not_ret_count = 0

        while not self.stop_event.is_set():
            if not self.is_running:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                    logger.debug(f"[{self.cam_name}] Stream released (paused)")
                self.stop_event.wait(0.5)
                continue

            if self.cap is None or not self.cap.isOpened():
                self._initialize_capture()
                if self.cap is None or not self.cap.isOpened():
                    self._handle_reconnection()
                    continue
                else:
                    self.reconnect_delay = 1.0

            ret, frame = self.cap.read()

            if ret:
                not_ret_count = 0
                with self.capture_lock:
                    self.capture_ret = True
                    self.latest_captured_frame = frame
                
                if not self.is_live_stream:
                    # In file mode, we wait for space in queue to avoid dropping frames
                    self.frame_queue.put(frame, block=True)
                
                self.new_captured_frame_event.set()
            else:
                if not self.is_live_stream:
                    self.stop_event.set()

                not_ret_count += 1
                with self.capture_lock:
                    self.capture_ret = False

                if not_ret_count > 10:
                    not_ret_count = 0
                    logger.warning(f"[{self.cam_name}] Stream broken, resetting...")
                    if self.cap is not None:
                        self.cap.release()
                    self.stop_event.wait(self.reconnect_delay)
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2,
                        self.max_reconnect_delay,
                    )
                else:
                    self.stop_event.wait(0.01)
                continue

        if self.cap is not None:
            self.cap.release()

        if self.ffmpeg_proc is not None:
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait(timeout=10)
            except Exception as e:
                logger.warning(f"[{self.cam_name}] FFmpeg cleanup error: {e}")
                self.ffmpeg_proc.kill()

    def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest processed frame as JPEG bytes (lazy encoding)"""
        with self.render_lock:
            if self.frame_to_render is None:
                return None
            frame = self.frame_to_render.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ret else None

    def _detection_loop(self):
        """Separate thread loop for heavy face detection"""
        logger.debug(f"[{self.cam_name}] Detection loop started")

        last_detection_time = 0
        min_interval = 1.0 / self.detect_fps

        while not self.stop_event.is_set():
            try:
                if self.new_frame_event.wait(timeout=1.0):
                    self.new_frame_event.clear()

                    current_time = time.time()
                    if (current_time - last_detection_time) < min_interval:
                        continue
                    last_detection_time = current_time

                    frame_to_process = None
                    with self.raw_frame_lock:
                        if self.latest_raw_frame is not None:
                            frame_to_process = self.latest_raw_frame.copy()

                    if frame_to_process is not None:
                        try:
                            result = self.sexual_harassment_detector.run(
                                frame_to_process
                            )
                        except Exception as e:
                            if not self.stop_event.is_set():
                                logger.error(f"[{self.cam_name}] Detection error: {e}")
                            result = None

                        with self.detections_lock:
                            self.current_detection = result

            except Exception as e:
                logger.error(f"[{self.cam_name}] Error in detection loop: {e}")

    def _process_frame(self, frame: MatLike):
        """Main frame processing pipeline: Resize -> Detect -> Render -> Track -> Save"""
        # 1. Provide to detector (Detector will handle resize in its run method)
        with self.raw_frame_lock:
            self.latest_raw_frame = frame
        self.new_frame_event.set()

        # 2. Get latest result (instant)
        result = None
        with self.detections_lock:
            result = self.current_detection

        # 3. Render
        frame_display = frame
        if result:
            frame_display = self.frame_renderer.render(frame, result)

        # 4. Update tracker
        self.sexual_harassment_tracker.update(frame_display, result)

        # 5. Save for lazy encoding
        with self.render_lock:
            self.frame_to_render = frame_display

        if self.ffmpeg_proc is not None:
            try:
                self.ffmpeg_proc.stdin.write(frame_display.tobytes())
            except BrokenPipeError:
                logger.warning(f"[{self.cam_name}] FFmpeg pipe closed unexpectedly")

    def run(self) -> None:
        """Main processing loop (consuming from capture thread)"""
        logger.info(f"[{self.cam_name}] Starting capture processor...")

        # Start threads
        threads = [
            threading.Thread(target=self._capture_loop, name=f"Capture_{self.cam_name}"),
            threading.Thread(target=self._detection_loop, name=f"Detection_{self.cam_name}"),
        ]
        for t in threads:
            t.start()

        while not self.stop_event.is_set():
            if not self.is_running:
                self.stop_event.wait(0.5)
                continue

            # 1. Get frame
            frame = None
            if self.is_live_stream:
                if not self.new_captured_frame_event.wait(timeout=1.0):
                    continue
                self.new_captured_frame_event.clear()

                with self.capture_lock:
                    if self.capture_ret and self.latest_captured_frame is not None:
                        frame = self.latest_captured_frame.copy()
            else:
                # File mode: pull from queue
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

            if frame is not None:
                self._process_frame(frame)

        # Cleanup
        logger.info(f"[{self.cam_name}] Stopping capture processor...")
        self.new_frame_event.set()

        for t in threads:
            if t.is_alive():
                t.join(timeout=2.0)
                if t.is_alive():
                    logger.warning(
                        f"[{self.cam_name}] Warning: {t.name} thread did not stop gracefully"
                    )

        try:
            cv2.destroyWindow(f"Stream: {self.cam_name}")
        except Exception:
            pass

        logger.info(f"[{self.cam_name}] Processor stopped")
