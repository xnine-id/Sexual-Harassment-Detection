from src.utils.config_loader import MQTTConfig
import json
import os
import logging
from datetime import datetime
from typing import Optional
from paho.mqtt import client as mqtt
from src.utils.config_loader import Config

logger = logging.getLogger("MQTT")


class MQTTService:
    """Centralized MQTT Service for publishing events, states, and handling commands"""

    def __init__(self, mqtt_config: MQTTConfig):
        self.mqtt_config = mqtt_config
        self.enabled = mqtt_config.enabled
        self.event_topic = mqtt_config.event_topic_prefix
        self.cmd_prefix = mqtt_config.command_topic_prefix
        self.state_prefix = mqtt_config.state_topic_prefix

        self.client = None
        self.command_callbacks = {}  # cam_name -> callback function

        if self.enabled:
            self._setup_client()

    def _setup_client(self):
        """Initialize and connect MQTT client"""
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        username = os.getenv("MQTT_AUTH_USERNAME")
        password = os.getenv("MQTT_AUTH_PASSWORD")
        host = os.getenv("MQTT_HOST", "localhost")
        port = int(os.getenv("MQTT_PORT", 1883))

        if username and password:
            self.client.username_pw_set(username, password)

        try:
            self.client.connect(host, port)
            self.client.loop_start()
            logger.info(f"[MQTT] Connected to {host}:{port}")
        except Exception as e:
            logger.exception(f"[MQTT] Connection failed: {e}")
            self.client = None

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection and resubscribe to topics"""
        logger.debug(f"[MQTT] Connected with result code {rc}")
        # Subscribe to wildcard command topic once
        self.client.subscribe(f"{self.cmd_prefix}/#")
        logger.info(f"[MQTT] Subscribed to {self.cmd_prefix}/#")

    def _on_message(self, client, userdata, msg):
        """Central message dispatcher"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic.startswith(self.cmd_prefix):
                cam_name = topic.replace(f"{self.cmd_prefix}/", "")
                if cam_name in self.command_callbacks:
                    self.command_callbacks[cam_name](payload)
        except Exception as e:
            logger.exception(f"[MQTT] Error handling message on {msg.topic}: {e}")

    def register_camera(self, cam_name, state, on_command_callback):
        """Register a camera for commands and initial state"""
        self.command_callbacks[cam_name] = on_command_callback

        if self.client:
            # Publish initial state (True by default)
            self.publish_state(cam_name, state)

    def publish_event(
        self,
        event_id: str,
        cam_name: str,
        confidence: float,
        snapshot: Optional[str] = None,
        event_type="harassment",
    ):
        """Publish detection event"""
        if not self.client:
            return

        payload = {
            "event_id": event_id,
            "camera": cam_name,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "snapshot": snapshot,
        }

        topic = f"{self.event_topic}/{cam_name}"
        self.client.publish(topic, json.dumps(payload), qos=1)

    def publish_state(self, cam_name, is_running):
        """Publish camera running state"""
        if not self.client:
            return

        topic = f"{self.state_prefix}/{cam_name}"
        self.client.publish(topic, json.dumps(is_running), retain=True)
        logger.debug(
            f"[MQTT] State: {cam_name} is {'RUNNING' if is_running else 'STOPPED'}"
        )

    def disconnect(self):
        """Cleanup connection"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("[MQTT] Disconnected")
