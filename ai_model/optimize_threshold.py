import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


# ==============================
# Configuration
# ==============================

BASE_DIR = r"C:\Users\naikp\Projects\AI_Defect_Detection_infosys"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_model",
    "bottle_autoencoder.pth"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test"
)


# ==============================
# Autoencoder Model
# ==============================

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ==============================
# Setup
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ==============================
# Calculate Anomaly Score
# ==============================

def get_score(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        reconstructed = model(image)

        loss = torch.mean(
            (image - reconstructed) ** 2
        )

    return loss.item()


# ==============================
# Collect Scores
# ==============================

normal_scores = []
defective_scores = []

defect_categories = [
    "broken_large",
    "broken_small",
    "contamination"
]


print("\nCalculating anomaly scores...\n")


# Normal images
good_dir = os.path.join(TEST_DIR, "good")

for filename in os.listdir(good_dir):

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(good_dir, filename)

        score = get_score(path)

        normal_scores.append(score)


# Defective images
for category in defect_categories:

    category_dir = os.path.join(TEST_DIR, category)

    if not os.path.exists(category_dir):
        continue

    for filename in os.listdir(category_dir):

        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            path = os.path.join(
                category_dir,
                filename
            )

            score = get_score(path)

            defective_scores.append(score)


print("Normal images:", len(normal_scores))
print("Defective images:", len(defective_scores))


# ==============================
# Threshold Optimization
# ==============================

all_scores = normal_scores + defective_scores

minimum = min(all_scores)
maximum = max(all_scores)

thresholds = np.linspace(
    minimum,
    maximum,
    200
)


best_threshold = None
best_f1 = -1

best_results = None


for threshold in thresholds:

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    # Normal images
    for score in normal_scores:

        if score >= threshold:
            FP += 1
        else:
            TN += 1


    # Defective images
    for score in defective_scores:

        if score >= threshold:
            TP += 1
        else:
            FN += 1


    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total


    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0


    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0


    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_results = (
            TP,
            TN,
            FP,
            FN,
            accuracy,
            precision,
            recall,
            f1
        )


# ==============================
# Results
# ==============================

TP, TN, FP, FN, accuracy, precision, recall, f1 = best_results


print("\n")
print("=============================================")
print("VISIONINSPECT AI - THRESHOLD OPTIMIZATION")
print("=============================================")

print(f"Best Threshold: {best_threshold:.6f}")

print("\nConfusion Matrix:")
print(f"True Positives:    {TP}")
print(f"True Negatives:    {TN}")
print(f"False Positives:   {FP}")
print(f"False Negatives:   {FN}")

print("\nPerformance:")
print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Precision:         {precision * 100:.2f}%")
print(f"Recall:            {recall * 100:.2f}%")
print(f"F1 Score:          {f1 * 100:.2f}%")

print("=============================================")