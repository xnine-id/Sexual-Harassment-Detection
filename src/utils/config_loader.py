import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DetectionSettings:
    model_path: str
    output_dir: str
    detect_fps: int = 5
    resize: List[int] = field(default_factory=lambda: [224, 224])


@dataclass
class SnapshotConfig:
    enabled: bool
    output_dir: str


@dataclass
class MQTTConfig:
    enabled: bool
    event_topic_prefix: str
    command_topic_prefix: str
    state_topic_prefix: str


@dataclass
class Config:
    detection_settings: DetectionSettings
    snapshot: SnapshotConfig
    mqtt: MQTTConfig


def validate_config(config: Dict[str, Any], required_keys: Dict[str, Any]) -> None:
    """Validate config structure recursively"""
    for key, expected in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

        if isinstance(expected, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"Config key '{key}' should be a dictionary")
            validate_config(config[key], expected)
        elif isinstance(expected, list):
            if not isinstance(config[key], list):
                raise ValueError(f"Config key '{key}' should be a list")
            if expected and isinstance(expected[0], dict):
                for item in config[key]:
                    validate_config(item, expected[0])


REQUIRED_CONFIG = {
    "detection_settings": {"model_path": str, "output_dir": str, "detect_fps": int, "resize": List[int]},
    "snapshot": {"enabled": bool, "output_dir": str},
    "mqtt": {
        "enabled": bool,
        "event_topic_prefix": str,
        "command_topic_prefix": str,
        "state_topic_prefix": str,
    },
}


def load_config(config_file: str) -> Config:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", config_file)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        validate_config(config_dict, REQUIRED_CONFIG)

    return Config(
        detection_settings=DetectionSettings(**config_dict["detection_settings"]),
        snapshot=SnapshotConfig(**config_dict["snapshot"]),
        mqtt=MQTTConfig(**config_dict["mqtt"]),
    )
