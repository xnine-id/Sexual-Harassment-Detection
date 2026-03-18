from cv2.typing import MatLike
import numpy as np
import logging
import cv2
from typing import Dict, Any
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

logger = logging.getLogger("SEXUAL_DETECTION")

class SexualHarassmentDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_settings = self.config["detection_settings"]
        self.base_model = VGG16(weights="imagenet", include_top=False)
        self.model = load_model(self.config["detection_settings"]["model_path"])

    def run(self, frame: MatLike):
        # Preprocessing
        preprocessed_frame = img_to_array(frame)
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
            "score": prob
        }

        return result
