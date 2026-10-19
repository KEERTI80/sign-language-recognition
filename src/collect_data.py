
import cv2
import os

# Dataset configuration
DATASET_PATH = "dataset"

SIGNS = SIGNS = [
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

IMAGES_PER_SIGN = 200

# Create folders for each sign
for sign in SIGNS:
    os.makedirs(os.path.join(DATASET_PATH, sign), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Webcam could not open.")
    raise SystemExit

cv2.namedWindow("Sign Language Dataset Collector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language Dataset Collector", 640, 480)

current_sign = None

print("Press 1-9 to select a sign.")
print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    # Display instructions
    cv2.putText(
        frame,
        f"Sign: {current_sign or 'Not selected'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "1-9: Select gesture | 0: Select gesture 10",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.imshow("Sign Language Dataset Collector", frame)

    key = cv2.waitKey(10) & 0xFF

    # Select sign
    if key in [ord(str(i)) for i in range(1, 10)]:
        current_sign = SIGNS[int(chr(key)) - 1]

    elif key == ord("0"):
        current_sign = SIGNS[9]

    if key in [ord(str(i)) for i in range(1, 10)] or key == ord("0"):
       print("Selected:", current_sign)


    # Capture image
    elif key == 32:
        if current_sign is None:
            print("Select a sign first.")
            continue

        folder = os.path.join(DATASET_PATH, current_sign)

        existing = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(existing) < IMAGES_PER_SIGN:
            image_path = os.path.join(
                folder,
                f"{len(existing) + 1}.jpg"
            )

            cv2.imwrite(image_path, frame)
            print(f"Saved {current_sign}: {len(existing) + 1}/200")
        else:
            print(f"{current_sign} already has 200 images.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Dataset collector closed.")
