import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ============================================================
# VISIONINSPECT AI - FULL DEFECT CLASSIFIER TEST
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_model",
    "bottle_defect_classifier.pth"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test"
)

CLASSES = [
    "broken_large",
    "broken_small",
    "contamination",
    "good"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("VISIONINSPECT AI - FULL DEFECT CLASSIFIER TEST")
print("=" * 65)
print(f"Device: {device}")

# ------------------------------------------------------------
# Load trained ResNet18
# ------------------------------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Trained classifier loaded successfully.")

# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

total = 0
correct = 0

confusion_matrix = {
    actual: {
        predicted: 0
        for predicted in CLASSES
    }
    for actual in CLASSES
}

# ------------------------------------------------------------
# Test all images
# ------------------------------------------------------------

for actual_class in CLASSES:

    class_folder = os.path.join(
        TEST_DIR,
        actual_class
    )

    if not os.path.exists(class_folder):
        print(f"Warning: folder not found: {class_folder}")
        continue

    image_files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    print()
    print(
        f"Testing {actual_class}: "
        f"{len(image_files)} images"
    )

    for image_file in sorted(image_files):

        image_path = os.path.join(
            class_folder,
            image_file
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():

                output = model(input_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = CLASSES[
                predicted.item()
            ]

            total += 1

            confusion_matrix[
                actual_class
            ][
                predicted_class
            ] += 1

            if predicted_class == actual_class:
                correct += 1

        except Exception as e:

            print(
                f"Error processing "
                f"{image_file}: {e}"
            )

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 65)
print("CLASSIFIER TEST RESULTS")
print("=" * 65)

print(f"Total images:     {total}")
print(f"Correct:          {correct}")
print(f"Incorrect:        {total - correct}")
print(f"Overall Accuracy: {accuracy:.2f}%")

print()
print("CONFUSION MATRIX")
print("-" * 65)

print(
    f"{'Actual':20s}"
    f"{'broken_large':15s}"
    f"{'broken_small':15s}"
    f"{'contamination':15s}"
    f"{'good':10s}"
)

for actual_class in CLASSES:

    print(
        f"{actual_class:20s}"
        f"{confusion_matrix[actual_class]['broken_large']:<15}"
        f"{confusion_matrix[actual_class]['broken_small']:<15}"
        f"{confusion_matrix[actual_class]['contamination']:<15}"
        f"{confusion_matrix[actual_class]['good']:<10}"
    )

# ------------------------------------------------------------
# Per-class accuracy
# ------------------------------------------------------------

print()
print("PER-CLASS ACCURACY")
print("-" * 65)

for class_name in CLASSES:

    actual_count = sum(
        confusion_matrix[class_name].values()
    )

    class_correct = confusion_matrix[
        class_name
    ][
        class_name
    ]

    class_accuracy = (
        class_correct / actual_count * 100
        if actual_count > 0
        else 0
    )

    print(
        f"{class_name:20s}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct}/{actual_count})"
    )

print()
print("=" * 65)
print("FULL CLASSIFIER TEST COMPLETE")
print("=" * 65)