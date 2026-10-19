
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os

# Load preprocessed data
data = np.load("processed_data.npz")
X = data["X"]
y = data["y"]
signs = data["signs"].tolist()

print("Dataset shape:", X.shape)
print("Classes:", signs)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Calculate class weights for imbalanced data
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes.tolist(), weights.tolist()))

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(signs), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on unseen test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification report:")
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(signs)),
    target_names=signs,
    zero_division=0
))

# Save trained model
os.makedirs("models", exist_ok=True)
model.save("models/sign_language_model.keras")

np.save("models/sign_labels.npy", np.array(signs))

print("\nModel saved successfully!")
print("Location: models/sign_language_model.keras")
