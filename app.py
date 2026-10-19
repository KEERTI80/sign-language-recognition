
import os

import mediapipe as mp
import numpy as np
import streamlit as st
import cv2
import time
import tensorflow as tf

from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SignAI | Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM WEBSITE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #111c30);
        color: #f4f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #101b2b;
        border-right: 1px solid #26384e;
    }

    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(120deg, #173b50, #29345f);
        border: 1px solid #36536b;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 44px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d0dfed;
        font-size: 17px;
    }

    .section-card {
        padding: 22px;
        border-radius: 16px;
        background: #17263a;
        border: 1px solid #30445c;
        margin-bottom: 15px;
    }

    .section-card h3 {
        color: #9be7d3;
    }

    .small-muted {
        color: #b5c5d8;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: #17263a;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #30445c;
    }

    div[data-testid="stMetricLabel"] {
        color: #c3d2e4;
    }

    div[data-testid="stMetricValue"] {
        color: #9be7d3;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    h2, h3 {
        color: #f4f7fb;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

MODEL_PATH = "models/sign_language_model.keras"
LABELS_PATH = "models/sign_labels.npy"
HAND_MODEL_PATH = "hand_landmarker.task"


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🤟 SignAI")
    st.caption("AI-powered gesture recognition")

    st.divider()

    page = st.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "📸 Recognize Gesture",
        "🎥 Live Recognition",
        "📚 About Project"
    ],
    label_visibility="visible"
)
    
    st.divider()

    st.markdown("### Project details")
    st.write("🧠 Deep Learning")
    st.write("✋ MediaPipe Landmarks")
    st.write("⚡ TensorFlow")
    st.write("🎯 10 Custom Gestures")

    st.divider()

    st.caption("Academic project prototype")


# --------------------------------------------------
# LOAD MODEL AND MEDIAPIPE
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(MODEL_PATH)

    labels = np.load(
        LABELS_PATH,
        allow_pickle=True
    ).tolist()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )

    landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    return model, labels, landmarker


required_files = [
    MODEL_PATH,
    LABELS_PATH,
    HAND_MODEL_PATH
]

if not all(os.path.isfile(path) for path in required_files):
    st.error(
        "A required model file is missing. "
        "Please check the models folder and hand_landmarker.task."
    )
    st.stop()

try:
    model, labels, landmarker = load_models()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Check the installed TensorFlow and MediaPipe versions "
        "if this error appears."
    )
    st.stop()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🤟 SignAI</h1>
        <p>Breaking barriers through hand gesture recognition.</p>
        <p>
            A computer vision and deep learning project that
            recognizes custom static hand gestures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Your recognition workspace")

    st.write(
        "Explore the project, capture a hand gesture, "
        "and view the model's prediction and confidence scores."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Gesture Classes", len(labels))

    with col2:
        st.metric("Input Features", model.input_shape[-1])

    with col3:
        st.metric("Recognition Mode", "Static")

    st.markdown("## How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div class="section-card">
            <h3>01 · Capture</h3>
            <p>Take a picture of your hand gesture.</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="section-card">
            <h3>02 · Extract</h3>
            <p>MediaPipe extracts 21 hand landmarks.</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="section-card">
            <h3>03 · Predict</h3>
            <p>The trained neural network predicts the gesture.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Choose 'Recognize Gesture' from the sidebar "
        "to test your trained model."
    )

    st.caption(
        "This prototype recognizes custom static gestures. "
        "It does not translate complete sign-language sentences."
    )


# --------------------------------------------------
# GESTURE RECOGNITION PAGE
# --------------------------------------------------

elif page == "📸 Recognize Gesture":

    st.markdown("## 📸 Gesture Recognition")
    st.write(
        "Position your hand clearly in the camera and capture a photo."
    )

    camera_col, result_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with camera_col:
        st.markdown("### Camera input")

        image_file = st.camera_input(
            "Capture your hand gesture",
            key="gesture_camera"
        )

    with result_col:
        st.markdown("### Prediction results")

        if image_file is None:
            st.info(
                "Your prediction will appear here "
                "after you capture a photo."
            )

            st.markdown("""
            <div class="section-card">
                <h3>Tips for better results</h3>
                <p>• Keep your whole hand visible.</p>
                <p>• Use clear, even lighting.</p>
                <p>• Keep your hand inside the camera frame.</p>
                <p>• Match the gestures used in your dataset.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            try:
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
                        landmarks.extend([
                            point.x,
                            point.y,
                            point.z
                        ])

                    features = np.array(
                        [landmarks],
                        dtype=np.float32
                    )

                    probabilities = model.predict(
                        features,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(probabilities)
                    )

                    predicted_sign = labels[predicted_index]
                    confidence = float(
                        probabilities[predicted_index]
                    )

                    st.success("Gesture prediction completed!")

                    st.metric(
                        "🤟 Predicted Gesture",
                        predicted_sign
                    )

                    st.metric(
                        "🎯 Model Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.progress(
                        confidence,
                        text="Confidence for the predicted class"
                    )

                else:
                    st.warning(
                        "No hand detected. Try again with "
                        "your hand clearly visible."
                    )

            except Exception as error:
                st.error(f"Could not process the image: {error}")

    # DISPLAY SCORES BELOW BOTH COLUMNS

    if image_file is not None:
        try:
            if "probabilities" in locals() and "predicted_sign" in locals():
                st.divider()
                st.markdown("### 📊 Scores for all gestures")

                sorted_indices = np.argsort(probabilities)[::-1]

                for index in sorted_indices:
                    sign = labels[index]
                    score = float(probabilities[index])

                    st.write(f"**{sign}** — {score * 100:.1f}%")
                    st.progress(score)

        except Exception:
            pass


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

# LIVE RECOGNITION PAGE

elif page == "🎥 Live Recognition":
    st.markdown("## 🎥 Live Gesture Recognition")
    st.write("Allow camera access to recognize your hand gestures.")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=HAND_MODEL_PATH
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1
                )
            )

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            result = self.landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                landmarks = []

                for point in hand:
                    landmarks.extend([point.x, point.y, point.z])

                features = np.array([landmarks], dtype=np.float32)
                probabilities = model.predict(features, verbose=0)[0]
                index = int(np.argmax(probabilities))
                sign = labels[index]
                confidence = float(probabilities[index])

                cv2.putText(
                    image,
                    f"{sign}: {confidence * 100:.1f}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    image,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(
        key="sign-language-live",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )





elif page == "📚 About Project":

    st.markdown("## 📚 About the Project")

    st.markdown("""
    <div class="hero">
        <h1>Sign Language Recognition</h1>
        <p>
            A deep learning prototype that classifies
            custom static hand gestures using hand landmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies used")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.markdown("""
        <div class="section-card">
            <h3>MediaPipe</h3>
            <p>Detects the hand and extracts 21 landmarks.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <h3>NumPy</h3>
            <p>Organizes the landmark coordinates as model input.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
        <div class="section-card">
            <h3>TensorFlow</h3>
            <p>Loads the trained model and predicts gesture classes.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <h3>Streamlit</h3>
            <p>Provides the interactive web interface.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Supported gesture labels")

    label_columns = st.columns(2)

    for index, sign in enumerate(labels):
        with label_columns[index % 2]:
            st.write(f"✋ {sign}")

    st.markdown("### Important limitation")

    st.warning(
        "This project recognizes the ten static gesture classes "
        "in its training labels. It does not recognize every sign "
        "language or translate continuous signing into sentences. "
        "Predictions depend on training data and image quality."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SignAI · Deep Learning & Computer Vision · Academic Prototype"
)
