from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "ai_model" / "bottle_autoencoder.pth"

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "test"
)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Same model architecture used during training
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Calculate anomaly score
def calculate_anomaly_score(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(image_tensor)
        error = torch.mean((image_tensor - reconstructed) ** 2)

    return error.item()


# Test images
print("Testing bottle images...\n")

for category in ["good", "broken_large", "broken_small", "contamination"]:
    category_path = TEST_DIR / category

    if not category_path.exists():
        continue

    image_files = list(category_path.glob("*.png"))

    if not image_files:
        image_files = list(category_path.glob("*.jpg"))

    for image_path in image_files[:5]:
        score = calculate_anomaly_score(image_path)

        print(
            f"Image: {image_path.name} | "
            f"Category: {category} | "
            f"Anomaly Score: {score:.6f}"
        )