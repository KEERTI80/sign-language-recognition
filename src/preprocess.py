
import os
import cv2
import numpy as np
import mediapipe as mp

# Project paths
DATASET_PATH = "dataset"
MODEL_PATH = "hand_landmarker.task"
OUTPUT_PATH = "processed_data.npz"

# Your 10 gesture classes
SIGNS = [
    "Hello",
    "Yes",
    "No",
    "Thank_You",
    "Stop",
    "OK",
    "Good",
    "Rock",
    "Three",
    "Point"
]

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_landmarks(image, landmarker):
    """Extract 21 hand landmarks, with x, y and z coordinates."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Use the first detected hand
    hand = result.hand_landmarks[0]

    landmarks = []
    for point in hand:
        landmarks.extend([point.x, point.y, point.z])

    return landmarks


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Place hand_landmarker.task in your project root.")
        return

    features = []
    labels = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    total_images = 0
    skipped_images = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_index, sign in enumerate(SIGNS):
            folder = os.path.join(DATASET_PATH, sign)

            if not os.path.isdir(folder):
                print(f"Folder missing: {folder}")
                continue

            image_names = sorted(os.listdir(folder))
            sign_count = 0

            for filename in image_names:
                image_path = os.path.join(folder, filename)

                image = cv2.imread(image_path)

                if image is None:
                    continue

                landmarks = extract_landmarks(image, landmarker)

                if landmarks is None:
                    skipped_images += 1
                    continue

                features.append(landmarks)
                labels.append(label_index)

                sign_count += 1
                total_images += 1

            print(f"{sign}: {sign_count} images processed")

    if len(features) == 0:
        print("\nNo hand landmarks detected.")
        print("Check your images and make sure hands are clearly visible.")
        return

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        signs=np.array(SIGNS)
    )

    print("\nPreprocessing complete!")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Images processed:", total_images)
    print("Images skipped (no hand detected):", skipped_images)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
