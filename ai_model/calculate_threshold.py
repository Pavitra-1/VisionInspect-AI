from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "ai_model" / "bottle_autoencoder.pth"

GOOD_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "test"
    / "good"
)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Same autoencoder architecture
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
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Calculate image score
def calculate_score(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(image_tensor)
        error = torch.mean((image_tensor - reconstructed) ** 2)

    return error.item()


# Calculate scores for normal images
scores = []

for image_path in GOOD_DIR.glob("*.png"):
    score = calculate_score(image_path)
    scores.append(score)

if not scores:
    raise RuntimeError("No normal test images found.")

# Mean + 3 standard deviations
mean_score = sum(scores) / len(scores)

variance = sum(
    (score - mean_score) ** 2 for score in scores
) / len(scores)

std_score = variance ** 0.5

threshold = mean_score + (3 * std_score)

print(f"Normal images tested: {len(scores)}")
print(f"Mean anomaly score: {mean_score:.6f}")
print(f"Standard deviation: {std_score:.6f}")
print(f"Suggested anomaly threshold: {threshold:.6f}")