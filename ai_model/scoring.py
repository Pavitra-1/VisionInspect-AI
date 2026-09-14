# ============================================
# VisionInspect AI - PDF Scoring Engine
# ============================================

def calculate_size_score(defect_area_ratio):

    percentage = max(
        0.0,
        min(100.0, defect_area_ratio * 100)
    )

    if percentage == 0:
        score = 0
    elif percentage < 0.5:
        score = 20
    elif percentage < 1.0:
        score = 40
    elif percentage < 2.0:
        score = 60
    elif percentage < 5.0:
        score = 80
    else:
        score = 100

    return round(score, 2)


def calculate_location_score(location_ratio):

    ratio = max(
        0.0,
        min(1.0, location_ratio)
    )

    return round(ratio * 100, 2)


def calculate_type_score(defect_type):

    scores = {
        "good": 0,
        "broken_small": 55,
        "contamination": 65,
        "broken_large": 95
    }

    return scores.get(
        defect_type,
        50
    )


def calculate_confidence(classifier_confidence):

    confidence = max(
        0.0,
        min(100.0, classifier_confidence)
    )

    return round(confidence, 2)


def calculate_severity(
    size_score,
    location_score,
    type_score,
    confidence_score
):

    severity = (
        size_score * 0.30
        + location_score * 0.25
        + type_score * 0.25
        + confidence_score * 0.20
    )

    return round(severity, 2)


def get_severity_level(severity_score):

    if severity_score >= 80:
        return "Critical"

    elif severity_score >= 60:
        return "High"

    elif severity_score >= 40:
        return "Medium"

    else:
        return "Low"


def get_recommended_action(severity_level):

    actions = {

        "Critical":
            "Reject product and trigger immediate quality inspection",

        "High":
            "Repair or rework recommended",

        "Medium":
            "Manual inspection and quality review required",

        "Low":
            "Product generally acceptable"
    }

    return actions.get(
        severity_level,
        "Manual inspection required"
    )


def get_pass_fail(severity_level):

    if severity_level in [
        "Critical",
        "High"
    ]:
        return "FAIL"

    return "PASS"


def generate_inspection_result(
    defect_type,
    defect_area_ratio,
    location_ratio,
    classifier_confidence
):

    # ============================================
    # GOOD PRODUCT
    # ============================================

    if defect_type == "good":

        confidence = calculate_confidence(
            classifier_confidence
        )

        return {

            "defect_type":
                "good",

            "defect_size_score":
                0,

            "defect_location_score":
                0,

            "defect_type_score":
                0,

            "detection_confidence":
                confidence,

            "severity_score":
                0,

            "severity_level":
                "Low",

            "recommended_action":
                "Product generally acceptable",

            "inspection_result":
                "PASS"
        }

    # ============================================
    # DEFECTIVE PRODUCT
    # ============================================

    size_score = calculate_size_score(
        defect_area_ratio
    )

    location_score = calculate_location_score(
        location_ratio
    )

    type_score = calculate_type_score(
        defect_type
    )

    confidence_score = calculate_confidence(
        classifier_confidence
    )

    severity_score = calculate_severity(
        size_score,
        location_score,
        type_score,
        confidence_score
    )

    severity_level = get_severity_level(
        severity_score
    )

    recommended_action = get_recommended_action(
        severity_level
    )

    result = get_pass_fail(
        severity_level
    )

    return {

        "defect_type":
            defect_type,

        "defect_size_score":
            size_score,

        "defect_location_score":
            location_score,

        "defect_type_score":
            type_score,

        "detection_confidence":
            confidence_score,

        "severity_score":
            severity_score,

        "severity_level":
            severity_level,

        "recommended_action":
            recommended_action,

        "inspection_result":
            result
    }
