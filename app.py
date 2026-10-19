
import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Capture a hand gesture to recognize it using deep learning.")

MODEL_PATH = "models/sign_language_model.keras"
LABELS_PATH = "models/sign_labels.npy"
HAND_MODEL_PATH = "hand_landmarker.task"


@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1
    )

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )

    return model, labels, landmarker


if not all(os.path.isfile(path) for path in [
    MODEL_PATH, LABELS_PATH, HAND_MODEL_PATH
]):
    st.error("A required model file is missing. Check your project folders.")
    st.stop()

try:
    model, labels, landmarker = load_models()
except Exception as error:
    st.error(f"Could not load the models: {error}")
    st.stop()

st.subheader("Capture your gesture")
image_file = st.camera_input("Take a picture of your hand")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    image_array = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmarks = []
        for point in hand:
            landmarks.extend([point.x, point.y, point.z])

        features = np.array([landmarks], dtype=np.float32)
        probabilities = model.predict(features, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_sign = labels[predicted_index]
        confidence = float(probabilities[predicted_index])

        st.success(f"Predicted gesture: {predicted_sign}")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")

        st.subheader("Scores for each gesture")
        for index, sign in enumerate(labels):
            st.write(f"{sign}: {probabilities[index] * 100:.1f}%")
    else:
        st.warning("No hand detected. Try again with your hand clearly visible.")

st.caption("This prototype recognizes 10 custom static hand gestures.")
