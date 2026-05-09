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

Features:
- Deep residual learning
- Skip connections
- Strong feature extraction
- Transfer learning support

---

## 2️⃣ ResNet50 + FPN

Feature Pyramid Network (FPN) improves feature representation for small-scale insect objects.

Advantages:
- Multi-scale feature extraction
- Better small object recognition
- Improved classification accuracy

---

## 3️⃣ Ensemble Learning

Soft Voting Ensemble combines predictions from:

- ResNet50
- ResNet50 + FPN

Final prediction is obtained using averaged probabilities.

Benefits:
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
- Dark modern UI
- GPU/CPU support

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
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- PIL
- NumPy

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/insect-pest-classification.git
cd insect-pest-classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📊 Dataset

The project is trained on the **IP102 Dataset**, a large-scale benchmark dataset for insect pest classification.

Dataset contains:
- Multiple insect pest categories
- Real-world agricultural images
- Fine-grained insect classes

---

# 📈 Results

| Model | Validation Accuracy |
|------|------|
| ResNet50 | ~68% |
| ResNet50 + FPN | ~66% |
| Ensemble | Improved Stability & Generalization |

---

# 👨‍💻 Students

- **Abdelrahman Saeed Abdelraoud** — 120220303
- **Youssef Ibrahim Mohammed** — 120220298
- **Mohamed Tareq Farouq** — 120220307
- **Hossam El Den Mahmoud** — 120220313
- **Mohamed Ahmed Abd Al Fatah** — 120220328
- **Ziad Reda** — 120220348

---

# 🏛️ Academic Information

Faculty of Computers and Artificial Intelligence  
Computer Vision Course  
Final Year Project

---

# 📷 Application Preview

<p align="center">
  <img src="EJUST.png" width="100%">
</p>

---

# 🔮 Future Improvements

- Add Residual Attention Networks (RAN)
- Add MMAL-Net
- Improve ensemble performance
- Mobile deployment
- Real-time camera inference
- Larger agricultural datasets

---

# 📜 License

This project is developed for academic and educational purposes.

---

# ⭐ Acknowledgments

Special thanks to:

- Faculty members
- Teaching assistants
- Open-source AI community
- PyTorch & Streamlit teams

---
