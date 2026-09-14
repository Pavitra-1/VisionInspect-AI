import cv2
import os

# MVTec AD dataset
dataset_path = "dataset/mvtec_anomaly_detection"

# Categories in MVTec AD
categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

# Process every category
for category in categories:

    input_folder = os.path.join(
        dataset_path,
        category,
        "train",
        "good"
    )

    output_folder = os.path.join(
        "dataset",
        "processed",
        category,
        "train"
    )

    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    if not os.path.exists(input_folder):
        print(f"⚠️ Training folder not found: {category}")
        continue

    for filename in os.listdir(input_folder):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            input_path = os.path.join(input_folder, filename)

            image = cv2.imread(input_path)

            if image is None:
                print("❌ Could not read:", input_path)
                continue

            # Resize
            resized = cv2.resize(image, (224, 224))

            # Convert to grayscale
            grayscale = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2GRAY
            )

            # Save processed image
            output_path = os.path.join(
                output_folder,
                filename
            )

            cv2.imwrite(output_path, grayscale)

            image_count += 1

    print(
        f"✅ {category}: "
        f"{image_count} training images processed"
    )

print("\n--------------------------------")
print("MVTec preprocessing completed!")
print("--------------------------------")