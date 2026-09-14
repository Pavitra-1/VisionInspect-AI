from backend.database import engine, Base, get_db
from backend import models

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials
)

from sqlalchemy.orm import Session

from backend.schemas import (
    UserCreate,
    UserResponse,
    LoginRequest
)

from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token
)

# =========================================================
# AI MODEL
# =========================================================

from ai_model.ai_inspector import inspect_image

import os
import cv2
import time
import numpy as np


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

Base.metadata.create_all(bind=engine)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="VisionInspect AI",
    description="Manufacturing Defect Detection & Quality Inspection System",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# JWT SECURITY
# =========================================================

security = HTTPBearer()


# =========================================================
# FOLDERS
# =========================================================

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "VisionInspect AI API is running",
        "version": "2.0.0",
        "ai_pipeline": (
            "Autoencoder + ResNet18 + Scoring"
        )
    }


# =========================================================
# IMAGE QUALITY ANALYSIS
# =========================================================

def analyze_image_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    quality_score = 100.0

    # -----------------------------------------------------
    # BRIGHTNESS
    # -----------------------------------------------------

    if brightness < 40 or brightness > 220:

        quality_score -= 25

    # -----------------------------------------------------
    # CONTRAST
    # -----------------------------------------------------

    if contrast < 20:

        quality_score -= 25

    # -----------------------------------------------------
    # SHARPNESS
    # -----------------------------------------------------

    if sharpness < 50:

        quality_score -= 20

    quality_score = max(
        0,
        min(
            100,
            quality_score
        )
    )

    # -----------------------------------------------------
    # QUALITY STATUS
    # -----------------------------------------------------

    if quality_score >= 70:

        quality_status = "Good"

    elif quality_score >= 40:

        quality_status = "Acceptable"

    else:

        quality_status = "Poor"

    return {

        "brightness":
            round(
                brightness,
                2
            ),

        "contrast":
            round(
                contrast,
                2
            ),

        "sharpness":
            round(
                sharpness,
                2
            ),

        "quality_score":
            round(
                quality_score,
                2
            ),

        "quality_status":
            quality_status
    }


# =========================================================
# LEGACY RULE-BASED DEFECT DETECTION
# =========================================================
#
# Kept for compatibility/reference.
#
# The /upload endpoint no longer uses this function.
# The actual inspection is performed by:
#
# Autoencoder + ResNet18 + scoring.py
#
# =========================================================

def detect_defect(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # NOISE REMOVAL
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # EDGE DETECTION
    # -----------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # -----------------------------------------------------
    # FIND CONTOURS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        image.shape[0]
        *
        image.shape[1]
    )

    significant_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area > image_area * 0.002:

            significant_contours.append(
                contour
            )

    # -----------------------------------------------------
    # NO SIGNIFICANT DEFECT
    # -----------------------------------------------------

    if not significant_contours:

        return {

            "defect_detected":
                False,

            "defect_type":
                "No Significant Defect",

            "defect_location":
                "None",

            "defect_area_pixels":
                0,

            "defect_area_percent":
                0,

            "bounding_box":
                None,

            "raw_confidence":
                0
        }

    # -----------------------------------------------------
    # LARGEST REGION
    # -----------------------------------------------------

    largest_contour = max(
        significant_contours,
        key=cv2.contourArea
    )

    defect_area = cv2.contourArea(
        largest_contour
    )

    defect_area_percent = (
        defect_area
        /
        image_area
    ) * 100

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    center_x = x + (w / 2)

    center_y = y + (h / 2)

    image_height, image_width = image.shape[:2]

    relative_x = (
        center_x /
        image_width
    )

    relative_y = (
        center_y /
        image_height
    )

    if (
        0.25 <= relative_x <= 0.75
        and
        0.25 <= relative_y <= 0.75
    ):

        location = (
            "Critical Functional Area"
        )

    elif (
        relative_x < 0.20
        or
        relative_x > 0.80
        or
        relative_y < 0.20
        or
        relative_y > 0.80
    ):

        location = (
            "Edge / Cosmetic Area"
        )

    else:

        location = "Surface Area"

    # -----------------------------------------------------
    # SIZE SCORE
    # -----------------------------------------------------

    size_score = min(
        100,
        max(
            0,
            defect_area_percent * 20
        )
    )

    # -----------------------------------------------------
    # TYPE HEURISTIC
    # -----------------------------------------------------

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4
            *
            np.pi
            *
            defect_area
        ) / (
            perimeter
            *
            perimeter
        )

    else:

        circularity = 0

    aspect_ratio = (
        max(w, h)
        /
        max(
            1,
            min(w, h)
        )
    )

    if aspect_ratio > 5:

        defect_type = "Scratch"

        type_score = 35

    elif circularity < 0.25:

        defect_type = (
            "Crack / Irregular Defect"
        )

        type_score = 85

    elif defect_area_percent > 5:

        defect_type = (
            "Major Surface Anomaly"
        )

        type_score = 80

    else:

        defect_type = (
            "Surface Anomaly"
        )

        type_score = 50

    # -----------------------------------------------------
    # LOCATION SCORE
    # -----------------------------------------------------

    if location == "Critical Functional Area":

        location_score = 90

    elif location == "Surface Area":

        location_score = 65

    else:

        location_score = 40

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = min(
        99,
        max(
            50,
            50
            +
            min(
                defect_area_percent * 5,
                25
            )
            +
            min(
                aspect_ratio * 3,
                15
            )
        )
    )

    return {

        "defect_detected":
            True,

        "defect_type":
            defect_type,

        "defect_location":
            location,

        "defect_area_pixels":
            round(
                defect_area,
                2
            ),

        "defect_area_percent":
            round(
                defect_area_percent,
                2
            ),

        "bounding_box": {

            "x":
                int(x),

            "y":
                int(y),

            "width":
                int(w),

            "height":
                int(h)
        },

        "size_score":
            round(
                size_score,
                2
            ),

        "location_score":
            round(
                location_score,
                2
            ),

        "type_score":
            round(
                type_score,
                2
            ),

        "confidence_score":
            round(
                confidence,
                2
            ),

        "raw_confidence":
            round(
                confidence,
                2
            )
    }


# =========================================================
# LEGACY SEVERITY SCORING
# =========================================================
#
# Kept for compatibility/reference.
#
# The /upload endpoint now receives the final scoring
# directly from ai_model/scoring.py through ai_inspector.py.
#
# =========================================================

def calculate_severity(
    size_score,
    location_score,
    type_score,
    confidence_score
):

    severity_score = (

        (size_score * 0.30)

        +

        (location_score * 0.25)

        +

        (type_score * 0.25)

        +

        (confidence_score * 0.20)
    )

    severity_score = round(
        severity_score,
        2
    )

    # -----------------------------------------------------
    # SEVERITY LEVEL
    # -----------------------------------------------------

    if severity_score >= 80:

        severity_level = "Critical"

        quality_decision = "FAIL"

        recommended_action = (
            "Reject product and trigger "
            "immediate quality inspection workflow"
        )

    elif severity_score >= 60:

        severity_level = "High"

        quality_decision = "FAIL"

        recommended_action = (
            "Repair or rework recommended"
        )

    elif severity_score >= 40:

        severity_level = "Medium"

        quality_decision = "MANUAL REVIEW"

        recommended_action = (
            "Perform additional inspection "
            "before final quality decision"
        )

    else:

        severity_level = "Low"

        quality_decision = "PASS"

        recommended_action = (
            "Minor cosmetic defect. "
            "Product generally acceptable"
        )

    return {

        "overall_severity_score":
            severity_score,

        "severity_level":
            severity_level,

        "quality_decision":
            quality_decision,

        "recommended_action":
            recommended_action
    }


# =========================================================
# IMAGE UPLOAD + AI INSPECTION
# =========================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    # =====================================================
    # FILE VALIDATION
    # =====================================================

    if (
        not file.content_type
        or
        not file.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid image file"
            )
        )

    # =====================================================
    # SAVE ORIGINAL IMAGE
    # =====================================================

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    file_content = await file.read()

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            file_content
        )

    # =====================================================
    # READ IMAGE
    # =====================================================

    image = cv2.imread(
        file_path
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded image"
            )
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # =====================================================
    # IMAGE QUALITY ANALYSIS
    # =====================================================

    image_quality = analyze_image_quality(
        image
    )

    # =====================================================
    # OPENCV PREPROCESSING
    # =====================================================
    #
    # This is retained for the preprocessing output and
    # processed-image visualization.
    #
    # The trained AI model receives the original uploaded
    # image through ai_inspector.py.
    #
    # =====================================================

    resized_image = cv2.resize(
        image,
        (224, 224)
    )

    gray_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2GRAY
    )

    denoised_image = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )

    enhanced_image = cv2.equalizeHist(
        denoised_image
    )

    # =====================================================
    # TRAINED AI INSPECTION
    # =====================================================
    #
    # IMPORTANT:
    #
    # Old:
    #     detect_defect(resized_image)
    #
    # New:
    #     inspect_image(file_path)
    #
    # The AI module performs:
    #
    #     Uploaded Image
    #          ↓
    #     Autoencoder
    #          ↓
    #     Anomaly Score
    #          ↓
    #     Defect Localization
    #          ↓
    #     ResNet18 Classifier
    #          ↓
    #     Defect Type + Confidence
    #          ↓
    #     scoring.py
    #          ↓
    #     Severity + PASS/FAIL
    #
    # =====================================================

    try:

        ai_result = inspect_image(
            file_path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI inspection failed: "
                + str(exc)
            )
        )

    # =====================================================
    # EXTRACT AI RESULTS
    # =====================================================

    ai_detection = ai_result[
        "ai_detection"
    ]

    anomaly_score = ai_result[
        "anomaly_score"
    ]

    detection_threshold = ai_result[
        "detection_threshold"
    ]

    defect_type = ai_result[
        "defect_type"
    ]

    classifier_confidence = ai_result[
        "classifier_confidence"
    ]

    defect_area_ratio = ai_result[
        "defect_area_ratio"
    ]

    location_ratio = ai_result[
        "location_ratio"
    ]

    scoring_result = ai_result[
        "scoring"
    ]

    # =====================================================
    # AI DETECTION BOOLEAN
    # =====================================================

    defect_detected = (
        ai_detection == "DEFECTIVE"
    )

    # =====================================================
    # SCORING PARAMETERS
    # =====================================================

    size_score = scoring_result[
        "defect_size_score"
    ]

    location_score = scoring_result[
        "defect_location_score"
    ]

    type_score = scoring_result[
        "defect_type_score"
    ]

    confidence_score = scoring_result[
        "detection_confidence"
    ]

    severity_score = scoring_result[
        "severity_score"
    ]

    severity_level = scoring_result[
        "severity_level"
    ]

    recommended_action = scoring_result[
        "recommended_action"
    ]

    inspection_result = scoring_result[
        "inspection_result"
    ]

    # =====================================================
    # DEFECT LOCATION DESCRIPTION
    # =====================================================

    if not defect_detected:

        defect_location = "None"

    elif location_ratio < 0.33:

        defect_location = "Upper Region"

    elif location_ratio < 0.66:

        defect_location = "Middle Region"

    else:

        defect_location = "Lower Region"

    # =====================================================
    # DEFECT AREA
    # =====================================================

    inspection_area_pixels = (
        128
        *
        128
    )

    defect_area_pixels = (
        defect_area_ratio
        *
        inspection_area_pixels
    )

    defect_area_percent = (
        defect_area_ratio
        *
        100
    )

    # =====================================================
    # BOUNDING BOX
    # =====================================================
    #
    # ai_inspector currently returns the defect area ratio
    # and location ratio rather than a final bounding box.
    #
    # Therefore we do not invent a bounding box here.
    #
    # =====================================================

    bounding_box = None

    # =====================================================
    # SAVE PROCESSED IMAGE
    # =====================================================

    processed_filename = (
        "processed_"
        +
        safe_filename
    )

    processed_path = os.path.join(
        PROCESSED_FOLDER,
        processed_filename
    )

    cv2.imwrite(
        processed_path,
        enhanced_image
    )

    # =====================================================
    # PROCESSING TIME
    # =====================================================

    processing_time = (
        time.perf_counter()
        -
        start_time
    )

    # =====================================================
    # COMPLETE INSPECTION REPORT
    # =====================================================

    return {

        # =================================================
        # GENERAL
        # =================================================

        "message":
            "Image inspection completed successfully",

        "inspection_status":
            "Completed",

        # =================================================
        # FILE INFORMATION
        # =================================================

        "filename":
            safe_filename,

        "original_path":
            file_path,

        "processed_path":
            processed_path,

        "original_image_size":
            f"{original_width}x{original_height}",

        "inspection_image_size":
            "128x128",

        # =================================================
        # AI MODEL PIPELINE
        # =================================================

        "ai_model": {

            "pipeline":
                (
                    "Autoencoder + "
                    "ResNet18 + Scoring"
                ),

            "autoencoder":
                "bottle_autoencoder.pth",

            "classifier":
                "bottle_defect_classifier.pth",

            "classifier_architecture":
                "ResNet18",

            "dataset":
                "MVTec AD - Bottle",

            "trained_model":
                True
        },

        # =================================================
        # PREPROCESSING
        # =================================================

        "preprocessing": {

            "resize":
                "128x128 for AI model",

            "dashboard_resize":
                "224x224",

            "grayscale":
                True,

            "noise_removal":
                "Gaussian Blur",

            "image_enhancement":
                "Histogram Equalization",

            "feature_extraction":
                (
                    "Autoencoder reconstruction "
                    "error + ResNet18 classification"
                )
        },

        # =================================================
        # IMAGE QUALITY
        # =================================================

        "image_quality":
            image_quality,

        # =================================================
        # AI DEFECT DETECTION
        # =================================================

        "defect_detection": {

            "detected":
                defect_detected,

            "ai_detection":
                ai_detection,

            "defect_type":
                defect_type,

            "defect_location":
                defect_location,

            "defect_area_pixels":
                round(
                    defect_area_pixels,
                    2
                ),

            "defect_area_percent":
                round(
                    defect_area_percent,
                    4
                ),

            "defect_area_ratio":
                defect_area_ratio,

            "location_ratio":
                location_ratio,

            "anomaly_score":
                anomaly_score,

            "detection_threshold":
                detection_threshold,

            "classifier_confidence":
                classifier_confidence,

            "bounding_box":
                bounding_box
        },

        # =================================================
        # SCORING PARAMETERS
        # =================================================

        "scoring": {

            "defect_size_score":
                round(
                    size_score,
                    2
                ),

            "defect_size_weight":
                "30%",

            "defect_location_score":
                round(
                    location_score,
                    2
                ),

            "defect_location_weight":
                "25%",

            "defect_type_score":
                round(
                    type_score,
                    2
                ),

            "defect_type_weight":
                "25%",

            "detection_confidence_score":
                round(
                    confidence_score,
                    2
                ),

            "detection_confidence_weight":
                "20%",

            "overall_severity_score":
                round(
                    severity_score,
                    2
                )
        },

        # =================================================
        # QUALITY ASSESSMENT
        # =================================================

        "quality_assessment": {

            "severity_level":
                severity_level,

            "quality_decision":
                inspection_result,

            "recommended_action":
                recommended_action
        },

        # =================================================
        # PERFORMANCE
        # =================================================

        "system_performance": {

            "processing_time_seconds":
                round(
                    processing_time,
                    4
                ),

            "inspection_response_time":
                f"{processing_time:.4f} seconds",

            "inspection_completed":
                True
        },

        # =================================================
        # AI MODEL METRICS
        # =================================================
        #
        # These require labeled MVTec AD evaluation.
        #
        # =================================================

        "model_performance": {

            "accuracy":
                None,

            "precision":
                None,

            "recall":
                None,

            "f1_score":
                None,

            "map_score":
                None,

            "evaluation_status":
                (
                    "Requires labeled MVTec AD "
                    "ground-truth evaluation"
                )
        },

        # =================================================
        # MANUFACTURING METRICS
        # =================================================

        "manufacturing_metrics": {

            "inspection_automation_rate":
                None,

            "defect_identification_accuracy":
                None,

            "false_defect_detection_rate":
                None,

            "evaluation_status":
                (
                    "Requires batch inspection "
                    "and ground-truth data"
                )
        }
    }


# =========================================================
# USER REGISTRATION
# =========================================================

@app.post(
    "/register",
    response_model=UserResponse
)
def register_user(
    user: UserCreate,
    db: Session = Depends(get_db)
):

    existing_user = db.query(
        models.User
    ).filter(
        models.User.email == user.email
    ).first()

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_user = models.User(

        name=user.name,

        email=user.email,

        hashed_password=hash_password(
            user.password
        ),

        role=user.role
    )

    db.add(
        new_user
    )

    db.commit()

    db.refresh(
        new_user
    )

    return new_user


# =========================================================
# USER LOGIN
# =========================================================

@app.post("/login")
def login_user(
    user: LoginRequest,
    db: Session = Depends(get_db)
):

    existing_user = db.query(
        models.User
    ).filter(
        models.User.email == user.email
    ).first()

    if not existing_user:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid email or password"
            )
        )

    password_correct = verify_password(
        user.password,
        existing_user.hashed_password
    )

    if not password_correct:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid email or password"
            )
        )

    access_token = create_access_token(

        data={

            "sub":
                str(
                    existing_user.id
                ),

            "email":
                existing_user.email,

            "role":
                existing_user.role
        }
    )

    return {

        "message":
            "Login successful",

        "access_token":
            access_token,

        "token_type":
            "bearer",

        "user": {

            "id":
                existing_user.id,

            "name":
                existing_user.name,

            "email":
                existing_user.email,

            "role":
                existing_user.role
        }
    }


# =========================================================
# PROTECTED PROFILE
# =========================================================

@app.get("/profile")
def get_profile(

    credentials:
        HTTPAuthorizationCredentials =
        Depends(security)
):

    token = credentials.credentials

    try:

        payload = decode_access_token(
            token
        )

        return {

            "message":
                "Authentication successful",

            "user_id":
                payload.get("sub"),

            "email":
                payload.get("email"),

            "role":
                payload.get("role")
        }

    except Exception:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or expired token"
            )
        )