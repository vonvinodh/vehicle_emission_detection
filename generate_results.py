import os
import shutil
from ultralytics import YOLO

# -------------------
# Paths
# -------------------
# Path to trained weights (update if different)
weights_path = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset\runs\detect\smoke_vehicle_detector\weights\best.pt"

# Path to test images
test_images = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset\test\images"

# Results folder
results_dir = r"C:\Users\Dell\Documents\vehicle_emission_detection\results"

# -------------------
# Prepare results folder
# -------------------
if os.path.exists(results_dir):
    print(f"[INFO] Removing old results folder: {results_dir}")
    shutil.rmtree(results_dir)

os.makedirs(results_dir, exist_ok=True)

# -------------------
# Load model & run inference
# -------------------
print("[INFO] Loading trained model...")
model = YOLO(weights_path)

print("[INFO] Running inference on test images...")
results = model.predict(
    source=test_images,
    save=True,
    project=results_dir,
    name=""  # prevent YOLO from creating subfolders
)

print(f"[INFO] Inference completed. Check results inside: {results_dir}")
