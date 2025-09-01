import os
import cv2
import hashlib
from glob import glob

# Path to your dataset
dataset_path = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset"
image_extensions = ['.jpg', '.jpeg', '.png']

# Counters
total_images = 0
resized_count = 0
missing_labels = 0
corrupt_removed = 0
duplicates_removed = 0

# Track duplicates by hash
hashes = set()

def get_file_hash(img_path):
    """Generate a hash for duplicate detection"""
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def process_image(img_path):
    """Check, resize, and detect duplicates"""
    global resized_count, corrupt_removed, duplicates_removed

    # Duplicate check
    img_hash = get_file_hash(img_path)
    if img_hash in hashes:
        os.remove(img_path)
        print(f" Duplicate removed: {img_path}")
        duplicates_removed += 1
        return False
    hashes.add(img_hash)

    # Check & resize
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f" Corrupt image removed: {img_path}")
            os.remove(img_path)
            corrupt_removed += 1
            return False

        if img.shape[0] != 640 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 640))
            cv2.imwrite(img_path, img)
            print(f" Resized: {img_path}")
            resized_count += 1

        return True

    except Exception as e:
        print(f" Error with {img_path}: {e}")
        os.remove(img_path)
        corrupt_removed += 1
        return False

def process_split(split):
    """Process images and labels in train/valid/test"""
    global total_images, missing_labels
    split_path = os.path.join(dataset_path, split)

    img_dir = os.path.join(split_path, "images")
    label_dir = os.path.join(split_path, "labels")

    if not os.path.exists(img_dir):
        print(f" {split} images folder not found, skipping.")
        return

    os.makedirs(label_dir, exist_ok=True)

    for img_path in glob(os.path.join(img_dir, "*")):
        ext = os.path.splitext(img_path)[-1].lower()
        if ext not in image_extensions:
            continue

        if not process_image(img_path):
            continue

        total_images += 1

        # Ensure label exists
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(label_dir, img_name + ".txt")

        if not os.path.exists(label_file):
            open(label_file, "w").close()
            print(f"Missing label created: {label_file}")
            missing_labels += 1


# Run preprocessing
for split in ["train", "valid", "test"]:
    print(f"\n Processing {split} dataset...")
    process_split(split)

# Summary
print("\nDataset preprocessing completed!")
print(" Summary:")
print(f"   Total images checked   : {total_images}")
print(f"   Images resized         : {resized_count}")
print(f"   Missing labels created : {missing_labels}")
print(f"   Corrupt images removed : {corrupt_removed}")
print(f"   Duplicates removed     : {duplicates_removed}")

# Save summary into a file
summary_path = os.path.join(dataset_path, "preprocess_summary.txt")
with open(summary_path, "w") as f:
    f.write(" Dataset preprocessing completed!\n")
    f.write(" Summary:\n")
    f.write(f"   Total images checked   : {total_images}\n")
    f.write(f"   Images resized         : {resized_count}\n")
    f.write(f"   Missing labels created : {missing_labels}\n")
    f.write(f"   Corrupt images removed : {corrupt_removed}\n")
    f.write(f"   Duplicates removed     : {duplicates_removed}\n")

print(f"\nSummary saved to: {summary_path}")
