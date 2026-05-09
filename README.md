<p align="center">
  <img src="EJUST.png" width="100%">
</p>

<h1 align="center">
Computer Vision Final Project
</h1>

<h3 align="center">
Insect Pest Classification using Deep Learning
</h3>

---

# 📌 Project Overview

This project presents an intelligent **Insect Pest Classification System** based on Deep Learning and Computer Vision techniques.

The system is capable of identifying insect pest species from uploaded images using advanced convolutional neural network architectures and ensemble learning methods.

The project includes:

- ResNet50 baseline model
- ResNet50 + Feature Pyramid Network (FPN)
- Soft Voting Ensemble
- Real-time inference using Streamlit
- Modern interactive user interface
- Deployment-ready architecture

---

# 🎯 Objectives

The main goals of this project are:

- Automate insect pest identification
- Improve agricultural monitoring systems
- Reduce manual classification effort
- Apply deep learning techniques in real-world computer vision tasks
- Build an end-to-end deployable AI application

---

# 🧠 Deep Learning Models

## 1️⃣ ResNet50

Residual Network (ResNet50) is used as the baseline model for image classification.

### Features
- Deep residual learning
- Skip connections
- Strong feature extraction
- Transfer learning support

---

## 2️⃣ ResNet50 + FPN

Feature Pyramid Network (FPN) improves feature representation for small-scale insect objects.

### Advantages
- Multi-scale feature extraction
- Better small object recognition
- Improved classification accuracy

---

## 3️⃣ Ensemble Learning

Soft Voting Ensemble combines predictions from:

- ResNet50
- ResNet50 + FPN

Final prediction is obtained using averaged probabilities.

### Benefits
- Better generalization
- Higher robustness
- Improved overall accuracy

---

# 🖥️ Streamlit Application

The project includes a professional Streamlit web application featuring:

- Real-time prediction
- Upload insect images
- Top-5 prediction results
- Confidence visualization
- Modern dark UI
- GPU/CPU support

---

# 🌐 Live Deployment

The application is deployed using Streamlit Cloud and can be accessed here:

🔗 **Live Demo:**  
https://abdelrahmansa04-insect-pest-classification-app-ap6y2v.streamlit.app/

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── requirements.txt
├── README.md
├── classes.txt
│
├── models/
│   ├── resnet50_best.pth
│   └── resnet50_fpn_best.pth
│
├── images/
│   └── college_logo.png
│
└── dataset/
