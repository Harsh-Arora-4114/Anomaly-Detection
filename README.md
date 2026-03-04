# Anomaly Detection

## Overview

This project focuses on detecting industrial product defects using Deep Learning and Unsupervised Learning techniques. The system is built using a Convolutional Autoencoder trained on the MVTec Anomaly Detection dataset.

The system integrates:

Convolutional Autoencoder – Deep learning reconstruction model
MVTec AD Dataset – Industrial defect dataset
Reconstruction Error (MSE) Evaluation
Automatic Threshold Selection (Mean + 3×Std)
Streamlit Interactive Dashboard

The application allows users to:

Detect defective industrial products
Visualize reconstructed images
Measure reconstruction error
Automatically classify Normal vs Anomaly
Adjust threshold dynamically
Analyze multiple industrial categories

This is a complete AI + Industrial Quality Inspection project suitable for research, academic submission, and resume portfolios.

---

## Objectives

Develop an AI-powered industrial defect detection system
Implement reconstruction-based anomaly detection
Train model only on normal samples (unsupervised learning)
Automatically compute anomaly threshold
Build an interactive industrial inspection dashboard
Demonstrate practical deep learning for manufacturing

---

## Key Features

### Industrial Defect Detection

Trained only on normal images
Detects scratches, cracks, dents, and structural defects
Works across multiple product categories

### Convolutional Autoencoder

Encoder: Conv2D + ReLU + MaxPooling
Decoder: ConvTranspose2D + ReLU
Loss Function: Mean Squared Error (MSE)
Learns reconstruction of normal patterns

### Automatic Threshold Calculation

Uses statistical formula:

threshold = mean_error + 3 × standard_deviation

Ensures reliable anomaly classification
Reduces manual tuning

### Model Evaluation

Reconstruction Error (MSE)
Threshold-based classification
Visual comparison: Original vs Reconstructed image

### Streamlit Dashboard

Upload custom images
View reconstruction results
Display anomaly score
Show automatic threshold
Real-time Normal / Anomaly decision

---

## Technologies Used

Category | Tools / Libraries
Programming Language | Python
Deep Learning | PyTorch
Web Framework | Streamlit
Data Processing | NumPy
Image Processing | Pillow, Torchvision
Visualization | Matplotlib
Dataset | MVTec Anomaly Detection

---

## Why Autoencoder?

Unsupervised learning approach
No defect labels required during training
Learns only normal data distribution
Defects produce high reconstruction error
Simple, effective, and industry applicable

---

## Project Workflow

### 1. Data Collection & Preparation

Use MVTec dataset
Select category (bottle, cable, toothbrush, etc.)
Train only on:

dataset/category/train/good

Resize images to fixed dimensions
Convert to tensor format

---

### 2. Model Training

Build Convolutional Autoencoder
Train on normal images
Optimize using MSE loss
Save trained model (autoencoder.pth)

---

### 3. Reconstruction Process

Input image → Autoencoder → Reconstructed image
Compute difference between original and output
Calculate reconstruction error

---

### 4. Anomaly Classification

If error ≤ threshold → Normal
If error > threshold → Anomaly

Threshold computed automatically using training error statistics

---

### 5. Dashboard Deployment

Run Streamlit app
Upload image
View classification result
Visualize reconstruction

---

## Sample Results

Input: Industrial product image

Outputs:

Reconstructed image
Reconstruction error value
Auto-calculated threshold
Final classification (Normal / Anomaly)

Key Insights:

Normal images produce low reconstruction error
Defective images produce high reconstruction error
Statistical threshold improves reliability

---

## How to Run

### 1. Clone Repository

git clone:https://github.com/Harsh-Arora-4114/Anomaly-Detection.git

---

### 2. Install Dependencies

pip install torch torchvision streamlit numpy pillow matplotlib

---

### 3. Train Model

python train_all.py

This generates:

autoencoder.pth

---

### 4. Run Application

python -m streamlit run app.py

Open browser:

[http://localhost:8501](http://localhost:8501)

---

## Dataset

Dataset Used: MVTec Anomaly Detection Dataset

Contains industrial categories such as:

bottle
,cable
,capsule
,toothbrush
,metal nut
,hazelnut
,grid
,wood
and more

Each category contains:

train/good
test/good
test/defect_types

---

## Future Scope

Add defect heatmap localization
Implement UNet-based architecture
Add ROC-AUC performance evaluation
Enable real-time camera inspection
Deploy on cloud (AWS / Streamlit Cloud)
Integrate industrial edge devices

---

## Applications

Industrial Quality Inspection
Smart Manufacturing Systems
Surface Defect Detection
AI-based Automation
Academic Deep Learning Research
Resume Project (ML Engineer / AI Engineer)

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by Harsh Arora
