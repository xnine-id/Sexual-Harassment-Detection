from src.utils.config_loader import DetectionSettings
import cv2
from cv2.typing import MatLike
import numpy as np
import logging
import os
from typing import Dict, Any
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from src.utils.model_downloader import download_model
import tensorflow as tf


logger = logging.getLogger("SEXUAL_DETECTION")


class SexualHarassmentDetector:
    def __init__(self, detection_settings: DetectionSettings):
        self.detection_settings = detection_settings
        
        # Check for GPU availability and log
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"SexualHarassmentDetector (Tensorflow) is using GPU: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.info("SexualHarassmentDetector (Tensorflow) is using CPU")

        self.base_model = VGG16(weights="imagenet", include_top=False)
        self.model = self.load_model()

        # Resize cache per instance (if needed, or move to a per-camera state)
        # Since the detector might be shared, it's better if preprocessing is pure
        # or we handle resizing here.
        self._resize_cache: Dict[str, Any] = {
            "input_shape": None,
            "new_size": None,
            "offsets": None,
            "canvas": None,
        }

    def load_model(self):
        if not os.path.exists(self.detection_settings.model_path):
            os.makedirs(os.path.dirname(self.detection_settings.model_path), exist_ok=True)
            download_model(os.getenv('MODEL_GDRIVE_ID'), self.detection_settings.model_path)

        return load_model(self.detection_settings.model_path)

    def preprocessing(self, frame: MatLike) -> MatLike:
        """Resize image to target dimensions while maintaining aspect ratio and padding with black"""
        target_width, target_height = self.detection_settings.resize
        h, w = frame.shape[:2]

        # Use cached values if input shape hasn't changed
        if self._resize_cache["input_shape"] != (h, w):
            ratio = min(target_width / w, target_height / h)
            new_w, new_h = int(w * ratio), int(h * ratio)

            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2

            self._resize_cache.update(
                {
                    "input_shape": (h, w),
                    "new_size": (new_w, new_h),
                    "offsets": (x_offset, y_offset),
                    "canvas": np.zeros(
                        (target_height, target_width, 3), dtype=np.uint8
                    ),
                }
            )

        new_w, new_h = self._resize_cache["new_size"]
        x_offset, y_offset = self._resize_cache["offsets"]

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = self._resize_cache["canvas"].copy()
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return canvas

    def run(self, frame: MatLike):
        # 1. Resize to target dimensions
        resized_frame = self.preprocessing(frame)

        # 2. VGG16 Preprocessing
        preprocessed_frame = img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = preprocess_input(preprocessed_frame)

        # Inference
        features = self.base_model.predict(preprocessed_frame, verbose=0)
        features_flatten = features.reshape(1, -1)
        prediction = self.model.predict(features_flatten, verbose=0)[0]

        class_label = np.argmax(prediction)
        prob = float(prediction[class_label])

        result = {
            "class": int(class_label),
            "label": "Harassment" if class_label == 1 else "Non-Harassment",
            "score": prob,
        }

        return result
