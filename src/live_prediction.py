
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model and gesture labels
model = tf.keras.models.load_model(
    "models/sign_language_model.keras"
)

signs = np.load("models/sign_labels.npy", allow_pickle=True).tolist()

# Set up MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# Open MacBook webcam
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Starting live prediction. Press Q to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam view
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # VIDEO mode requires a timestamp in milliseconds
        timestamp_ms = int(
            cv2.getTickCount() / cv2.getTickFrequency() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            landmarks = []
            for point in hand:
                landmarks.extend([point.x, point.y, point.z])

            # Predict the gesture
            input_data = np.array(
                [landmarks], dtype=np.float32
            )

            probabilities = model.predict(
                input_data, verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            predicted_sign = signs[predicted_index]

            # Display prediction
            text = f"{predicted_sign}: {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Draw hand landmark points
            height, width, _ = frame.shape

            for point in hand:
                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display webcam window
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

