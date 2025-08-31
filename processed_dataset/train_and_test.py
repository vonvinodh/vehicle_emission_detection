from ultralytics import YOLO

# --------- STEP 1: Define dataset path ---------
data_yaml = r"C:\Users\Dell\Downloads\project_dataset\processed_dataset\data.yaml"

# --------- STEP 2: Train the model ---------
print(" Starting Training...")
model = YOLO("yolov8n.pt")   # you can change to yolov8s.pt for better accuracy

model.train(
    data=data_yaml,          # dataset yaml path
    epochs=50,               # number of epochs
    imgsz=640,               # image size
    batch=16,                # batch size
    name="smoke_vehicle_detector"  # experiment name
)

print(" Training Completed!")

# --------- STEP 3: Evaluate model on test set ---------
print("Evaluating on test set...")
metrics = model.val(data=data_yaml, split="test")
print("Evaluation Results:", metrics)

# --------- STEP 4: Run inference on sample image/video ---------
# Replace path with your own test file
sample_file = r"C:\Users\Dell\Downloads\car_video.mp4"  # can be .jpg, .png, or video

print(f" Running inference on: {sample_file}")
results = model.predict(source=sample_file, show=True, save=True)

print(" Done! Results saved inside 'runs/detect/smoke_vehicle_detector/predict'")
