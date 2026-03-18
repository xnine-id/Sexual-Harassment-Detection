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


def predict_and_display_camera(
    model,
    base_model,
    url: Union[str, int] = 0,
    detect_fps: int = 5,
    confidence_threshold: float = 0.5,
):
    cap = cv2.VideoCapture(url)

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps // detect_fps)

    # Try to set buffer size for RTSP to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error(f"Cannot open video source: {url}")
        return

    st.write("Live Stream:")
    stream = st.empty()

    # Use a session state to control the loop
    if "run" not in st.session_state:
        st.session_state.run = True

    frame_count = 0
    last_label = "Scanning..."
    last_prob = 0.0
    last_color = (0, 255, 0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame received or stream ended.")
            break

        frame_count += 1

        # Only perform inference every `skip_frames` frames
        if frame_count % skip_frames == 0:
            # Preprocessing
            resized_frame = cv2.resize(frame, (224, 224))
            preprocessed_frame = img_to_array(resized_frame)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
            preprocessed_frame = preprocess_input(preprocessed_frame)

            # Feature extraction (heavy task)
            features = base_model.predict(preprocessed_frame, verbose=0)
            features_flatten = features.reshape(1, -1)

            # Classification
            prediction = model.predict(features_flatten, verbose=0)[0]
            class_label = np.argmax(prediction)
            prob = prediction[class_label]

            # Apply confidence threshold
            if prob >= confidence_threshold:
                last_label = "Harassment" if class_label == 1 else "Non-Harassment"
                last_color = (0, 0, 255) if class_label == 1 else (0, 255, 0)
                last_prob = prob
            else:
                last_label = "Uncertain"
                last_color = (255, 255, 0)  # Yellow
                last_prob = prob

        # Overlay the *last* known prediction on *every* frame
        prob_text = f"{last_label} ({last_prob:.2f})"
        cv2.putText(
            frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2
        )

        # Display the frame
        stream.image(
            frame,
            channels="BGR",
            caption=f"Live Prediction (Inference every {skip_frames} frames)",
        )

        # Small sleep to keep UI responsive
        time.sleep(0.001)

    cap.release()


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
