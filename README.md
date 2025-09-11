# Vehicle Emission Detection using YOLOv8
## 📌 Project Overview
Air pollution from vehicle exhaust is one of the leading causes of urban air quality degradation.
This project uses YOLOv8 (You Only Look Once) object detection to identify vehicles emitting visible smoke from street images.

The system:
1. Collects and preprocesses vehicle + smoke datasets.
2. Trains a YOLOv8 model to detect smoke emissions.
3. Evaluates model performance.
4. Runs inference on new images and outputs detection results.


### 📂 Project Structure
vehicle_emission_detection/
│
├── preprocess_check.py        # Verify dataset integrity, labels, resizing
├── process_dataset.py         # Data cleaning & preprocessing
├── train_and_test.py          # Training, validation, and test pipeline
├── generate_results.py        # Run inference and save predictions
│
├── processed_dataset/         # Dataset (train/valid/test + data.yaml)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── results/                   # Inference outputs (created automatically)
│
└── runs/                      # Training logs, weights, metrics (auto-generated)


### ⚙️ Installation
1. Clone the repo:
git clone https://github.com/<your-username>/vehicle_emission_detection.git
cd vehicle_emission_detection
2. Install dependencies:
pip install ultralytics opencv-python matplotlib


#### 🚀 Usage
1. Preprocess Dataset
Checks for missing labels, resizes images, and removes corrupt files:
python preprocess_check.py

2. Train & Test Model
Trains YOLOv8 on the dataset (by default uses a small fraction for demo):
python train_and_test.py

Outputs:
- runs/smoke_vehicle_detector/weights/best.pt → trained model
- Training metrics saved in runs/

3. Run Inference
Generates predictions on test images and saves results:
python generate_results.py

Results are saved in:
results/


#### 📊 Results
1. Training Metrics
The training progress is visualized in the curves below (loss decreasing, mAP improving).
[Training Results](docs/results.png)  

2. Confusion Matrix
This shows how well the model distinguishes smoke vs no-smoke cases.
[Confusion Matrix](docs/confusion_matrix.png)  

3. Sample Predictions
The model outputs bounding boxes with confidence scores around detected smoke emissions.


he model outputs bounding boxes with confidence scores around detected smoke emissions.  

| ✅ Correct Detection              |  ⚠️ False Positive            |
|-----------------------             |-------------------            |
| ![Correct](docs/correct_pred.jpg)  | ![False](docs/false_pred.jpg) | 


- Correct Detection → Vehicle emitting visible smoke is detected successfully.
- False Positive → A bounding box is predicted even though no actual smoke is visible (e.g., shadows/exhaust pipes).



#### ✅ Summary:
- The model successfully detects vehicle smoke in many cases.
- False positives occur, which can be reduced with more data and fine-tuning.


#### ⚠️ Limitations
- False positives: Sometimes detects smoke where none exists (shadows, exhaust pipes).
- Small dataset: Current demo uses ~5–10% of total data.
- Few epochs: Demo trains only for 1–2 epochs for speed.


##### 🔮 Future Improvements
- Train on full dataset with more epochs (20–50).
- Improve dataset quality (remove noisy labels, add balanced samples).
- Deploy as a real-time CCTV monitoring system.
- Integrate with traffic police systems for alerts.


###### 📚 References
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Vehicle smoke datasets (public sources, custom labeling)

### 📂 Dataset & Weights  

Due to size limitations, the **full dataset** and **trained model weights** are stored on Google Drive.  

- Dataset (train/valid/test): [Google Drive Link](https://drive.google.com/drive/folders/1aooAJAPcpiz38u84VTy5oAD1eGP56un4?usp=drive_link)  
- Trained YOLOv8 model (best.pt): [Google Drive Link](https://drive.google.com/file/d/1FMuJU4ao-f9d_f8GmT95W5yn2o1rloBc/view?usp=drive_link)  


