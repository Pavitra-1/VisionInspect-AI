import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms, models

from .scoring import generate_inspection_result


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 32, 3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64, 3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128, 3,
                stride=2,
                padding=1
            ),
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


# ============================================================
# LOAD RESNET18 CLASSIFIER
# ============================================================

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
# TRANSFORMS
# ============================================================

autoencoder_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


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


# ============================================================
# ANOMALY THRESHOLD
# ============================================================

THRESHOLD = 0.001560


# ============================================================
# MAIN AI INSPECTION FUNCTION
# ============================================================

def inspect_image(image_path):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # AUTOENCODER
    # --------------------------------------------------------

    input_tensor = (
        autoencoder_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        reconstructed = autoencoder(
            input_tensor
        )

        error_map = torch.mean(
            (
                input_tensor -
                reconstructed
            ) ** 2,
            dim=1
        )

        anomaly_score = (
            error_map.mean().item()
        )


    # --------------------------------------------------------
    # DEFECT LOCALIZATION
    # --------------------------------------------------------

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

    normalized_error = (
        normalized_error.astype(
            np.uint8
        )
    )


    adaptive_threshold = max(
        30,
        min(
            100,
            float(
                np.mean(
                    normalized_error
                )
                +
                2.5 *
                np.std(
                    normalized_error
                )
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


    # --------------------------------------------------------
    # DEFECT SIZE
    # --------------------------------------------------------

    total_area = (
        defect_mask.shape[0]
        *
        defect_mask.shape[1]
    )

    defect_area = float(
        np.count_nonzero(
            defect_mask
        )
    )

    defect_area_ratio = (
        defect_area /
        total_area
    )


    # --------------------------------------------------------
    # DEFECT LOCATION
    # --------------------------------------------------------

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

            center_y = (
                y +
                (h / 2)
            )

            normalized_y = (
                center_y /
                defect_mask.shape[0]
            )

            if normalized_y < 0.33:

                location_ratio = 0.30

            elif normalized_y < 0.66:

                location_ratio = 0.60

            else:

                location_ratio = 0.80


    # --------------------------------------------------------
    # RESNET18 CLASSIFICATION
    # --------------------------------------------------------

    classifier_input = (
        classifier_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = classifier(
            classifier_input
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    defect_type = CLASS_NAMES[
        predicted_index.item()
    ]

    classifier_confidence = (
        confidence.item() *
        100
    )


    # --------------------------------------------------------
    # FINAL AI DECISION
    # --------------------------------------------------------

    if defect_type == "good":

        final_detection = "NORMAL"

    else:

        final_detection = "DEFECTIVE"


    # --------------------------------------------------------
    # PDF SCORING
    # --------------------------------------------------------

    scoring_result = (
        generate_inspection_result(
            defect_type=defect_type,
            defect_area_ratio=defect_area_ratio,
            location_ratio=location_ratio,
            classifier_confidence=classifier_confidence
        )
    )


    # --------------------------------------------------------
    # RETURN JSON-SAFE RESULT
    # --------------------------------------------------------

    return {

        "ai_detection":
            final_detection,

        "anomaly_score":
            round(
                float(anomaly_score),
                6
            ),

        "detection_threshold":
            THRESHOLD,

        "defect_type":
            defect_type,

        "classifier_confidence":
            round(
                float(
                    classifier_confidence
                ),
                2
            ),

        "defect_area_ratio":
            round(
                float(
                    defect_area_ratio
                ),
                6
            ),

        "location_ratio":
            round(
                float(
                    location_ratio
                ),
                4
            ),

        "scoring":
            scoring_result
    }