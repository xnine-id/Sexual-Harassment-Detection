import yaml
import os

from typing import Dict, Any


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


def load_config(config_file: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required = {
        "detection_settings": {"model_path": str, "detection_threshold": float, "output_dir": str},
        "cameras": [{"name": str, "url": str, "enabled": bool}],
        "snapshot": {"enabled": bool, "output_dir": str},
        "mqtt": {
            "enabled": bool,
            "event_topic_prefix": str,
            "command_topic_prefix": str,
            "state_topic_prefix": str,
        },
    }
    validate_config(config, required)
    return config
