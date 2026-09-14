import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================
# VisionInspect AI
# Defect Region Detection
# ============================================

PROJECT_ROOT = r"C:\Users\naikp\Projects\AI_Defect_Detection_infosys"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai_model",
    "bottle_autoencoder.pth"
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "mvtec_anomaly_detection",
    "bottle",
    "test",
    "broken_large",
    "000.png"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "defect_detection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# Device
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================
# Autoencoder
# Same architecture used during training
# ============================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3,
                kernel_size=3,
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


# ============================================
# Load trained AI model
# ============================================

model = Autoencoder().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Trained model loaded successfully.")


# ============================================
# Image preprocessing
# ============================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================
# Load test image
# ============================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

input_tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================
# AI reconstruction
# ============================================

with torch.no_grad():

    reconstruction = model(
        input_tensor
    )


# ============================================
# Convert tensors to NumPy
# ============================================

original = (
    input_tensor
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)

reconstructed = (
    reconstruction
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)


# ============================================
# Pixel reconstruction error
# ============================================

difference = np.abs(
    original - reconstructed
)

difference = np.mean(
    difference,
    axis=2
)


# ============================================
# Anomaly score
# ============================================

anomaly_score = float(
    difference.mean()
)

print()
print("======================================")
print("AI ANOMALY ANALYSIS")
print("======================================")

print(
    "Anomaly score:",
    round(anomaly_score, 6)
)

print(
    "Maximum pixel error:",
    round(float(difference.max()), 6)
)


# ============================================
# Convert error map to 0-255
# ============================================

difference_uint8 = cv2.normalize(
    difference,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ============================================
# Heatmap
# ============================================

heatmap = cv2.applyColorMap(
    difference_uint8,
    cv2.COLORMAP_JET
)


# ============================================
# Adaptive threshold
# ============================================

mean_value = float(
    difference_uint8.mean()
)

std_value = float(
    difference_uint8.std()
)

threshold_value = (
    mean_value +
    1.5 * std_value
)

threshold_value = max(
    20,
    min(200, threshold_value)
)

print(
    "Adaptive defect threshold:",
    round(threshold_value, 2)
)


# ============================================
# Defect mask
# ============================================

_, mask = cv2.threshold(
    difference_uint8,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)


# ============================================
# Remove noise
# ============================================

kernel = np.ones(
    (3, 3),
    np.uint8
)

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


# ============================================
# Find defect regions
# ============================================

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ============================================
# Filter very small regions
# ============================================

filtered_contours = []

for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area >= 5:

        filtered_contours.append(
            contour
        )


# ============================================
# Calculate defect area
# ============================================

total_image_area = (
    mask.shape[0] *
    mask.shape[1]
)

defect_area = sum(
    cv2.contourArea(c)
    for c in filtered_contours
)

defect_area_ratio = (
    defect_area /
    total_image_area
)

defect_area_percentage = (
    defect_area_ratio *
    100
)


# ============================================
# Largest defect
# ============================================

largest_contour = None
largest_area = 0

for contour in filtered_contours:

    area = cv2.contourArea(
        contour
    )

    if area > largest_area:

        largest_area = area
        largest_contour = contour


# ============================================
# Draw detected defects
# ============================================

result_image = cv2.resize(
    np.array(image),
    (128, 128)
)

for contour in filtered_contours:

    x, y, w, h = cv2.boundingRect(
        contour
    )

    cv2.rectangle(
        result_image,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )


# ============================================
# Save results
# ============================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "heatmap.png"
    ),
    heatmap
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "defect_mask.png"
    ),
    mask
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "detected_defect.png"
    ),
    cv2.cvtColor(
        result_image,
        cv2.COLOR_RGB2BGR
    )
)


# ============================================
# Final output
# ============================================

print()
print("======================================")
print("DEFECT REGION ANALYSIS")
print("======================================")

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Total image area:",
    total_image_area
)

print(
    "Detected defect area:",
    round(defect_area, 2)
)

print(
    "Defect area ratio:",
    round(
        defect_area_ratio,
        4
    )
)

print(
    "Defect area percentage:",
    round(
        defect_area_percentage,
        2
    ),
    "%"
)

print(
    "Number of defect regions:",
    len(filtered_contours)
)

if largest_contour is not None:

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    print(
        "Largest defect area:",
        round(largest_area, 2)
    )

    print(
        "Largest defect location:",
        f"x={x}, y={y}"
    )

    print(
        "Largest defect size:",
        f"{w} x {h}"
    )

else:

    print(
        "No significant defect region detected."
    )


print()
print("Output files saved to:")

print(
    os.path.join(
        OUTPUT_DIR,
        "heatmap.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "defect_mask.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "detected_defect.png"
    )
)

print("======================================")