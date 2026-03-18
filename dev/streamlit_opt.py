from typing import Union
import streamlit as st
import cv2
import numpy as np
import gdown  # To download the model from Google Drive
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# Function to load and cache the model
@st.cache_resource
def load_cached_model(file_id, output_path):
    # gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    model = load_model(output_path)
    return model


# Function to load and cache the base VGG16 model
@st.cache_resource
def load_base_model():
    return VGG16(weights="imagenet", include_top=False)


import time
import threading


class VideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def predict_and_display_camera(
    model,
    base_model,
    url: Union[str, int] = 0,
    detect_fps: int = 5,
    confidence_threshold: float = 0.5,
):
    vs = VideoStream(url).start()

    if not vs.ret:
        st.error(f"Cannot open video source: {url}")
        vs.stop()
        return

    st.write("Live Stream (Zero Latency Mode):")
    stream = st.empty()

    if "run" not in st.session_state:
        st.session_state.run = True

    last_label = "Scanning..."
    last_prob = 0.0
    last_color = (0, 255, 0)

    # Calculate inference interval in seconds
    inference_interval = 1.0 / detect_fps
    last_inference_time = 0

    while st.session_state.run:
        ret, frame = vs.read()
        if not ret or vs.stopped:
            st.warning("No frame received or stream ended.")
            break

        current_time = time.time()

        # Only perform inference based on time interval instead of frame count
        # This is more accurate for real-time
        if (current_time - last_inference_time) > inference_interval:
            last_inference_time = current_time

            # Preprocessing (copy frame to avoid issues with thread)
            img = frame.copy()
            resized_frame = cv2.resize(img, (224, 224))
            preprocessed_frame = img_to_array(resized_frame)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
            preprocessed_frame = preprocess_input(preprocessed_frame)

            # Inference
            features = base_model.predict(preprocessed_frame, verbose=0)
            features_flatten = features.reshape(1, -1)
            prediction = model.predict(features_flatten, verbose=0)[0]

            class_label = np.argmax(prediction)
            prob = prediction[class_label]

            if prob >= confidence_threshold:
                last_label = "Harassment" if class_label == 1 else "Non-Harassment"
                last_color = (0, 0, 255) if class_label == 1 else (0, 255, 0)
                last_prob = prob
            else:
                last_label = "Uncertain"
                last_color = (255, 255, 0)
                last_prob = prob

        # Overlay & Display
        display_frame = frame.copy()
        prob_text = f"{last_label} ({last_prob:.2f})"
        cv2.putText(
            display_frame,
            prob_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            last_color,
            2,
        )

        stream.image(
            display_frame,
            channels="BGR",
            caption=f"Real-time Stream | Detection Rate: {detect_fps} FPS",
        )

        # Very small sleep to yield to other threads
        time.sleep(0.001)

    vs.stop()


def main():
    st.title("Live Harassment Detection")

    # Configuration Sidebar
    st.sidebar.header("Configuration")
    detect_fps = st.sidebar.slider(
        "Inference Rate (Frames per Second)",
        min_value=1,
        max_value=30,
        value=5,
        help="Only run AI detection every N frames. Higher = Faster/Lower Latency.",
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show classification if probability is above this value.",
    )

    # Google Drive file ID
    file_id = "1GP2IdE-mPdQ9D3ouDIqbIU-3gl26Kgf2"

    # Path to where the model is located
    model_path = "weight.hdf5"

    # Load cached models
    try:
        model = load_cached_model(file_id, model_path)
        base_model = load_base_model()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Prediction"):
            st.session_state.run = True
            predict_and_display_camera(
                model,
                base_model,
                "rtsp://localhost:8554/input/webcam1",
                detect_fps=detect_fps,
                confidence_threshold=confidence_threshold,
            )

    with col2:
        if st.button("Stop"):
            st.session_state.run = False
            st.write("Stopping...")


if __name__ == "__main__":
    main()
