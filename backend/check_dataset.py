import os

dataset_path = "dataset/mvtec_anomaly_detection"

# MVTec AD categories
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

print("✅ MVTec AD dataset found!\n")

total_images = 0

for category in categories:
    category_path = os.path.join(dataset_path, category)

    train_path = os.path.join(category_path, "train")
    test_path = os.path.join(category_path, "test")

    train_count = 0
    test_count = 0

    # Count training images
    if os.path.exists(train_path):
        for root, dirs, files in os.walk(train_path):
            train_count += len([
                f for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

    # Count test images
    if os.path.exists(test_path):
        for root, dirs, files in os.walk(test_path):
            test_count += len([
                f for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

    category_total = train_count + test_count
    total_images += category_total

    print(
        f"{category:12} "
        f"Train: {train_count:4} | "
        f"Test: {test_count:4} | "
        f"Total: {category_total:4}"
    )

print("\n--------------------------------")
print("Total images:", total_images)
print("--------------------------------")