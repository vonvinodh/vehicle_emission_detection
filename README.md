# Vehicle Emission Detection using YOLOv8  

##  Project Overview  
Air pollution from vehicle exhaust is one of the leading causes of urban air quality degradation.  
This project uses **YOLOv8 (You Only Look Once)** object detection to **identify vehicles emitting visible smoke** from street images.  

The system:  
1. Collects and preprocesses vehicle + smoke datasets.  
2. Trains a YOLOv8 model to detect smoke emissions.  
3. Evaluates model performance.  
4. Runs inference on new images and outputs detection results.  

---

##  Project Structure  

vehicle_emission_detection/
│
├── preprocess_check.py # Verify dataset integrity, labels, resizing
├── process_dataset.py # Data cleaning & preprocessing
├── train_and_test.py # Training, validation, and test pipeline
├── generate_results.py # Run inference and save predictions
│
├── processed_dataset/ # Dataset (train/valid/test + data.yaml)
│ ├── train/
│ ├── valid/
│ ├── test/
│ └── data.yaml
│
├── results/ # Inference outputs (created automatically)
│
└── runs/ # Training logs, weights, metrics (auto-generated)


---

##  Installation  

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/vehicle_emission_detection.git
cd vehicle_emission_detection


Install dependencies:

pip install ultralytics opencv-python matplotlib

 Usage
1. Preprocess Dataset

Checks for missing labels, resizes images, and removes corrupt files:

python preprocess_check.py

2. Train & Test Model

Trains YOLOv8 on the dataset (by default uses a small fraction for demo):

python train_and_test.py


Outputs:

runs/smoke_vehicle_detector/weights/best.pt → trained model

Training metrics in runs/

3. Run Inference

Generates predictions on test images and saves results:

python generate_results.py


Results saved in:

results/

 Results

The model detects smoke emission from vehicles in road images.

Example outputs (bounding boxes with confidence scores):

Image	Detection

	
 Limitations

False positives: Sometimes detects smoke where none exists (shadows, exhaust pipes).

Small dataset: Current demo uses ~5–10% of total data.

Few epochs: Demo trains only for 1–2 epochs for speed.

 Future Improvements

Train on full dataset with more epochs (20–50).

Improve dataset quality (remove noisy labels, add balanced samples).

Deploy as a real-time CCTV monitoring system.

Integrate with traffic police systems for alerts.