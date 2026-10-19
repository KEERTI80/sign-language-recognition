
import os

DATASET_PATH = "dataset"

SIGNS = [
    "Hello", "Yes", "No", "Thank_You", "Stop",
    "OK", "Good", "Rock", "Three", "Point"
]

total_images = 0
counts = []

print("\nSIGN LANGUAGE DATASET REPORT")
print("-" * 45)

for sign in SIGNS:
    folder = os.path.join(DATASET_PATH, sign)

    if not os.path.isdir(folder):
        print(f"{sign:12} | Folder missing")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)
    total_images += count
    counts.append(count)

    status = "LOW DATA" if count < 20 else "OK"
    print(f"{sign:12} | {count:4} images | {status}")

print("-" * 45)
print("Total images:", total_images)

if counts:
    print("Average per gesture:", round(sum(counts) / len(counts), 1))
    print("Smallest class:", min(counts))
    print("Largest class:", max(counts))
