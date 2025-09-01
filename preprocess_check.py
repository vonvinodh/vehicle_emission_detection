import os
import cv2
from glob import glob

# Path to your dataset
dataset_path = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset"
image_extensions = ['.jpg', '.jpeg', '.png']

# Counters
total_images = 0
resized_count = 0
missing_labels = 0
corrupt_removed = 0

def verify_and_process_image(img_path):
    """Check image validity and resize if not 640x640"""
    global resized_count, corrupt_removed
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f" Corrupt image removed: {img_path}")
            os.remove(img_path)
            corrupt_removed += 1
            return

        if img.shape[0] != 640 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 640))
            cv2.imwrite(img_path, img)
            print(f"Resized: {img_path}")
            resized_count += 1

    except Exception as e:
        print(f" Error with {img_path}: {e}")
        os.remove(img_path)
        corrupt_removed += 1

def check_split(split):
    """Check train/valid/test dataset split"""
    global total_images, missing_labels
    split_path = os.path.join(dataset_path, split)

    # Expect subfolders like train/images, train/labels
    img_dir = os.path.join(split_path, "images")
    label_dir = os.path.join(split_path, "labels")

    if not os.path.exists(img_dir):
        print(f"{split} images folder not found, skipping.")
        return

    os.makedirs(label_dir, exist_ok=True)

    for img_path in glob(os.path.join(img_dir, "*")):
        ext = os.path.splitext(img_path)[-1].lower()
        if ext not in image_extensions:
            continue

        total_images += 1
        verify_and_process_image(img_path)

        # Ensure label exists
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(label_dir, img_name + ".txt")

        if not os.path.exists(label_file):
            open(label_file, "w").close()
            print(f" Missing label created: {label_file}")
            missing_labels += 1


# Run checks
for split in ["train", "valid", "test"]:
    print(f"\n Checking {split} dataset...")
    check_split(split)

# Summary
print("\n Preprocessing & verification completed!")
print(" Summary:")
print(f"   Total images checked   : {total_images}")
print(f"   Images resized         : {resized_count}")
print(f"   Missing labels created : {missing_labels}")
print(f"   Corrupt images removed : {corrupt_removed}")
