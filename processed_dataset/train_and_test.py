from ultralytics import YOLO
import os

# Paths
dataset_path = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset\data.yaml"
output_dir = r"C:\Users\Dell\Documents\vehicle_emission_detection\processed_dataset\runs"

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model (nano version for faster CPU training)
model = YOLO("yolov8n.pt")

print("\n Starting Training...")
model.train(
    data=dataset_path,       # your dataset yaml
    epochs=1,               # keep small for demo
    imgsz=640,
    batch=8,                 # small batch size for CPU
    fraction=0.05,           # use only 25% of dataset to save time
    device="cpu",            # you don’t have GPU
    project=output_dir,
    name="smoke_vehicle_detector"
)
print("\nTraining Finished!")

# Evaluate on validation set
print("\n Running Evaluation...")
metrics = model.val(data=dataset_path)
print(metrics)

# Run predictions on test images
print("\n Running Inference on Test Images...")
results = model.predict(
    source=os.path.join(os.path.dirname(dataset_path), "test/images"), 
    save=True,
    project=output_dir,
    name="predictions"
)

print("\n Inference Completed! Check results inside 'runs/predictions'.")
