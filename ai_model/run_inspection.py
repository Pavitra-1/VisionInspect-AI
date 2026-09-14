import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms, models

from scoring import generate_inspection_result


PROJECT_ROOT = r"C:\Users\naikp\Projects\AI_Defect_Detection_infosys"


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "broken_large",
    "000.png"
)


# ============================================================
# MODEL PATHS
# ============================================================

AUTOENCODER_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_model",
    "bottle_autoencoder.pth"
)

CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_model",
    "bottle_defect_classifier.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3, 3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(
            self.encoder(x)
        )


# ============================================================
# LOAD AUTOENCODER
# ============================================================

autoencoder = Autoencoder().to(DEVICE)

autoencoder.load_state_dict(
    torch.load(
        AUTOENCODER_PATH,
        map_location=DEVICE
    )
)

autoencoder.eval()

print("Trained autoencoder loaded.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# AUTOENCODER PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

input_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(DEVICE)
)


# ============================================================
# AUTOENCODER ANOMALY ANALYSIS
# ============================================================

with torch.no_grad():

    reconstructed = autoencoder(
        input_tensor
    )

    error_map = torch.mean(
        (input_tensor - reconstructed) ** 2,
        dim=1
    )

    anomaly_score = (
        error_map.mean().item()
    )


# ============================================================
# PROTOTYPE THRESHOLD
# ============================================================

THRESHOLD = 0.001560


# ============================================================
# DEFECT LOCALIZATION
# ============================================================

error_array = (
    error_map
    .squeeze()
    .cpu()
    .numpy()
)

normalized_error = cv2.normalize(
    error_array,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

normalized_error = normalized_error.astype(
    np.uint8
)


adaptive_threshold = max(
    30,
    min(
        100,
        float(
            np.mean(normalized_error)
            +
            2.5 * np.std(normalized_error)
        )
    )
)


_, defect_mask = cv2.threshold(
    normalized_error,
    adaptive_threshold,
    255,
    cv2.THRESH_BINARY
)


kernel = np.ones(
    (3, 3),
    np.uint8
)

defect_mask = cv2.morphologyEx(
    defect_mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# DEFECT SIZE
# ============================================================

total_area = (
    defect_mask.shape[0]
    *
    defect_mask.shape[1]
)

defect_area = float(
    np.count_nonzero(defect_mask)
)

defect_area_ratio = (
    defect_area / total_area
)


# ============================================================
# DEFECT LOCATION
# ============================================================

location_ratio = 0.0

contours, _ = cv2.findContours(
    defect_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    largest_area = cv2.contourArea(
        largest_contour
    )

    if largest_area > 0:

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        center_y = y + (h / 2)

        normalized_y = (
            center_y
            /
            defect_mask.shape[0]
        )

        if normalized_y < 0.33:
            location_ratio = 0.30

        elif normalized_y < 0.66:
            location_ratio = 0.60

        else:
            location_ratio = 0.80


# ============================================================
# LOAD RESNET18 CLASSIFIER
# ============================================================

print()
print("Loading trained defect classifier...")

classifier = models.resnet18(
    weights=None
)

classifier.fc = nn.Linear(
    classifier.fc.in_features,
    4
)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE
    )
)

classifier = classifier.to(DEVICE)

classifier.eval()

print(
    "Trained ResNet18 classifier loaded."
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "broken_large",
    "broken_small",
    "contamination",
    "good"
]


# ============================================================
# CLASSIFIER PREPROCESSING
# ============================================================

classifier_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


classifier_input = (
    classifier_transform(image)
    .unsqueeze(0)
    .to(DEVICE)
)


# ============================================================
# AI CLASSIFICATION
# ============================================================

with torch.no_grad():

    logits = classifier(
        classifier_input
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )


defect_type = CLASS_NAMES[
    predicted_index.item()
]

classifier_confidence = (
    confidence.item() * 100
)


# ============================================================
# FINAL AI DECISION
# ============================================================

if defect_type == "good":

    final_detection = "NORMAL"

else:

    final_detection = "DEFECTIVE"


# ============================================================
# PDF SCORING
# ============================================================

result = generate_inspection_result(
    defect_type=defect_type,
    defect_area_ratio=defect_area_ratio,
    location_ratio=location_ratio,
    classifier_confidence=classifier_confidence
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("============================================")
print("VISIONINSPECT AI - INSPECTION RESULT")
print("============================================")

print(
    f"Image: {IMAGE_PATH}"
)

print()
print(
    f"AI Detection: {final_detection}"
)

print(
    f"Anomaly Score: {anomaly_score:.6f}"
)

print(
    f"Detection Threshold: {THRESHOLD}"
)

print()
print("--------------------------------------------")
print("AI DEFECT CLASSIFICATION")
print("--------------------------------------------")

print(
    f"AI Predicted Defect Type: {defect_type}"
)

print(
    f"AI Classifier Confidence: "
    f"{classifier_confidence:.2f} %"
)

print()
print("--------------------------------------------")
print("PDF SCORING PARAMETERS")
print("--------------------------------------------")

print(
    f"Defect Type: {defect_type}"
)

print(
    f"Defect Size Score (30%): "
    f"{result['defect_size_score']}"
)

print(
    f"Defect Location Score (25%): "
    f"{result['defect_location_score']}"
)

print(
    f"Defect Type Score (25%): "
    f"{result['defect_type_score']}"
)

print(
    f"Detection Confidence (20%): "
    f"{result['detection_confidence']}"
)

print()
print("--------------------------------------------")
print("FINAL QUALITY ASSESSMENT")
print("--------------------------------------------")

print(
    f"Severity Score: "
    f"{result['severity_score']}"
)

print(
    f"Severity Level: "
    f"{result['severity_level']}"
)

print(
    f"Inspection Result: "
    f"{result['inspection_result']}"
)

print(
    f"Recommended Action: "
    f"{result['recommended_action']}"
)

print("============================================")

