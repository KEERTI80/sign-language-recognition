# sign-language-recognition
Sign Language Recognition using Deep Learning, TensorFlow and MediaPipe
# Sign Language Recognition Using Deep Learning

A deep learning project that recognizes 10 custom static hand gestures using MediaPipe hand landmarks and a TensorFlow neural network.

## Features

- Hand landmark detection using MediaPipe.
- Gesture classification using TensorFlow.
- Webcam-based gesture prediction.
- Streamlit web interface for capturing and classifying gestures.
- Confidence scores for predicted gestures.

## Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Streamlit

## Supported Gestures

Hello, Yes, No, Thank You, Stop, OK, Good, Rock, Three, and Point.

## Project Structure

```text
sign-language-recognition/
├── app.py
├── requirements.txt
├── hand_landmarker.task
├── models/
│   ├── sign_language_model.keras
│   └── sign_labels.npy
└── src/
    ├── collect_data.py
    ├── inspect_dataset.py
    ├── preprocess.py
    ├── train_model.py
    └── live_prediction.py
```

## How to Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the web app:

   `streamlit run app.py`

## Model Performance

The current model achieved **83.87% test accuracy** on a held-out test set of 93 samples.

Performance may vary depending on lighting, hand position, and background conditions.

## Limitations

This prototype recognizes 10 custom static hand gestures. It does not yet translate complete sign language, including gestures that depend on movement or facial expressions.

## Future Improvements

- Expand the dataset with more varied examples.
- Improve accuracy for confusing gestures.
- Add support for more gesture classes.
- Improve the user interface and real-time recognition.

## Disclaimer

This is an educational prototype, not a complete sign language translation system.
