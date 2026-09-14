from pathlib import Path
import csv

import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "ai_model"
    / "bottle_autoencoder.pth"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "milestone2_evaluation.csv"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ============================================================
# AUTOENCODER MODEL
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 3, 2, 1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, 3, 2, 1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3, 3, 2, 1,
                output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Trained AI model loaded successfully.")


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 0.002782


# ============================================================
# DEFECT CATEGORIES
# ============================================================

DEFECT_CATEGORIES = [
    "broken_large",
    "broken_small",
    "contamination"
]


# ============================================================
# TYPE SCORES
# Based on your PDF scoring categories
# ============================================================

TYPE_SCORES = {
    "good": 0,
    "broken_small": 55,
    "contamination": 65,
    "broken_large": 95,
    "unknown": 50
}


# ============================================================
# AI INFERENCE
# ============================================================

def analyze_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        reconstructed = model(image_tensor)

        error_map = torch.mean(
            (image_tensor - reconstructed) ** 2,
            dim=1
        )

        anomaly_score = torch.mean(
            (image_tensor - reconstructed) ** 2
        ).item()

    error_map = error_map.squeeze(0).cpu().numpy()

    return anomaly_score, error_map


# ============================================================
# DEFECT REGION DETECTION
# ============================================================

def detect_defect_region(error_map):

    # Normalize reconstruction error
    normalized = cv2.normalize(
        error_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized = normalized.astype(np.uint8)

    # Adaptive threshold
    threshold_value = np.percentile(
        normalized,
        99
    )

    _, mask = cv2.threshold(
        normalized,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = 128 * 128

    total_defect_area = 0
    largest_area = 0
    largest_center_y = 64
    number_of_regions = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 2:
            continue

        number_of_regions += 1
        total_defect_area += area

        if area > largest_area:

            largest_area = area

            moments = cv2.moments(contour)

            if moments["m00"] != 0:
                center_y = (
                    moments["m01"]
                    / moments["m00"]
                )

                largest_center_y = center_y

    defect_area_ratio = (
        total_defect_area / image_area
    )

    return (
        defect_area_ratio,
        largest_area,
        largest_center_y,
        number_of_regions
    )


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def calculate_size_score(defect_area_ratio):

    score = defect_area_ratio * 100

    return min(score, 100)


def calculate_location_score(center_y):

    # Prototype location zones:
    #
    # Upper region    -> 30
    # Middle region   -> 60
    # Lower region   -> 80
    #
    # This will be improved later using
    # product-specific ROI zones.

    location_ratio = center_y / 128

    if location_ratio < 0.33:
        return 30.0

    elif location_ratio < 0.66:
        return 60.0

    else:
        return 80.0


def calculate_type_score(actual_type):

    return TYPE_SCORES.get(
        actual_type,
        TYPE_SCORES["unknown"]
    )


def calculate_confidence(
    anomaly_score,
    threshold
):

    if anomaly_score <= 0:
        return 0.0

    # Relative anomaly strength
    ratio = anomaly_score / threshold

    confidence = min(
        ratio * 50,
        100
    )

    return confidence


# ============================================================
# SEVERITY CALCULATION
# ============================================================

def calculate_severity(
    size_score,
    location_score,
    type_score,
    confidence
):

    severity = (
        (size_score * 0.30)
        +
        (location_score * 0.25)
        +
        (type_score * 0.25)
        +
        (confidence * 0.20)
    )

    return severity


def get_severity_level(score):

    if score >= 80:
        return "Critical"

    elif score >= 60:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


def get_pass_fail(level):

    if level in ["Critical", "High"]:
        return "FAIL"

    return "PASS"


def get_recommended_action(level):

    if level == "Critical":
        return (
            "Reject product and trigger immediate "
            "quality inspection"
        )

    elif level == "High":
        return "Repair or rework recommended"

    elif level == "Medium":
        return (
            "Manual inspection and quality review required"
        )

    else:
        return "Product generally acceptable"


# ============================================================
# EVALUATION COUNTERS
# ============================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

total_normal = 0
total_defective = 0

results = []


# ============================================================
# ALL TEST CATEGORIES
# ============================================================

categories = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]


# ============================================================
# EVALUATE ALL 83 IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("VISIONINSPECT AI - MILESTONE 2 FULL EVALUATION")
print("=" * 70)

for category in categories:

    category_dir = TEST_DIR / category

    if not category_dir.exists():
        print(
            f"WARNING: Category not found: {category}"
        )
        continue

    image_paths = sorted(
        category_dir.glob("*.png")
    )

    print(
        f"\nEvaluating {category}: "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        # ----------------------------------------------------
        # AI inference
        # ----------------------------------------------------

        anomaly_score, error_map = analyze_image(
            image_path
        )

        prediction_defective = (
            anomaly_score >= THRESHOLD
        )

        ai_prediction = (
            "DEFECTIVE"
            if prediction_defective
            else "GOOD"
        )

        # ----------------------------------------------------
        # Classification metrics
        # ----------------------------------------------------

        actual_defective = (
            category != "good"
        )

        if actual_defective:
            total_defective += 1

            if prediction_defective:
                true_positive += 1
            else:
                false_negative += 1

        else:
            total_normal += 1

            if prediction_defective:
                false_positive += 1
            else:
                true_negative += 1

        # ----------------------------------------------------
        # Defect region
        # ----------------------------------------------------

        if prediction_defective:

            (
                defect_area_ratio,
                largest_area,
                center_y,
                number_of_regions
            ) = detect_defect_region(
                error_map
            )

        else:

            defect_area_ratio = 0
            largest_area = 0
            center_y = 64
            number_of_regions = 0

        # ----------------------------------------------------
        # Milestone 2 scoring
        # ----------------------------------------------------

        size_score = calculate_size_score(
            defect_area_ratio
        )

        location_score = calculate_location_score(
            center_y
        )

        type_score = calculate_type_score(
            category
        )

        confidence = calculate_confidence(
            anomaly_score,
            THRESHOLD
        )

        severity_score = calculate_severity(
            size_score,
            location_score,
            type_score,
            confidence
        )

        severity_level = get_severity_level(
            severity_score
        )

        inspection_result = get_pass_fail(
            severity_level
        )

        action = get_recommended_action(
            severity_level
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "Image":
                image_path.name,

            "Actual Type":
                category,

            "AI Prediction":
                ai_prediction,

            "Anomaly Score":
                round(anomaly_score, 6),

            "Defect Area":
                round(defect_area_ratio * 16384, 2),

            "Defect Area %":
                round(defect_area_ratio * 100, 3),

            "Defect Regions":
                number_of_regions,

            "Defect Size Score":
                round(size_score, 2),

            "Defect Location Score":
                round(location_score, 2),

            "Defect Type Score":
                type_score,

            "Detection Confidence":
                round(confidence, 2),

            "Severity Score":
                round(severity_score, 2),

            "Severity Level":
                severity_level,

            "Inspection Result":
                inspection_result,

            "Recommended Action":
                action
        })


# ============================================================
# MODEL METRICS
# ============================================================

total_images = (
    total_normal +
    total_defective
)

accuracy = (
    (true_positive + true_negative)
    / total_images
    if total_images > 0
    else 0
)

precision = (
    true_positive
    / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# SAVE CSV REPORT
# ============================================================

if results:

    fieldnames = list(
        results[0].keys()
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Normal images:      {total_normal}")
print(f"Defective images:   {total_defective}")
print(f"Total images:       {total_images}")

print("\nConfusion Matrix:")

print(f"True Positives:     {true_positive}")
print(f"True Negatives:     {true_negative}")
print(f"False Positives:    {false_positive}")
print(f"False Negatives:    {false_negative}")

print("\nPerformance:")

print(
    f"Accuracy:           {accuracy * 100:.2f}%"
)

print(
    f"Precision:          {precision * 100:.2f}%"
)

print(
    f"Recall:             {recall * 100:.2f}%"
)

print(
    f"F1 Score:           {f1_score * 100:.2f}%"
)


# ============================================================
# SAMPLE INSPECTION RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE MILESTONE 2 INSPECTION RESULTS")
print("=" * 70)

for result in results[:10]:

    print("\nImage:", result["Image"])

    print(
        "Actual Type:",
        result["Actual Type"]
    )

    print(
        "AI Prediction:",
        result["AI Prediction"]
    )

    print(
        "Anomaly Score:",
        result["Anomaly Score"]
    )

    print(
        "Defect Area:",
        result["Defect Area"],
        "pixels"
    )

    print(
        "Defect Size Score:",
        result["Defect Size Score"]
    )

    print(
        "Defect Location Score:",
        result["Defect Location Score"]
    )

    print(
        "Defect Type Score:",
        result["Defect Type Score"]
    )

    print(
        "Detection Confidence:",
        result["Detection Confidence"]
    )

    print(
        "Severity Score:",
        result["Severity Score"]
    )

    print(
        "Severity Level:",
        result["Severity Level"]
    )

    print(
        "Inspection Result:",
        result["Inspection Result"]
    )

    print(
        "Recommended Action:",
        result["Recommended Action"]
    )


print("\n")
print("=" * 70)
print("MILESTONE 2 EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Full report saved to:\n{CSV_PATH}"
)

print("=" * 70)