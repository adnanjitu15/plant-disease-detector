---
title: Plant Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🌿 Plant Disease Detector

**Professional AI Model with 97.78% Accuracy** - Detect 39 different plant diseases instantly!

## 🚀 Live Demo
Upload a plant leaf image and get instant disease diagnosis with confidence scores.

## 📊 Model Performance
- **Validation Accuracy:** 97.78%
- **Classes:** 39 plant diseases
- **Training Data:** 55,448 images
- **Architecture:** EfficientNet-B0
- **Training Time:** 17.5 minutes on Tesla T4 GPU

## 🌱 Supported Plants & Diseases
- **Apple:** Scab, Black Rot, Cedar Rust, Healthy
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Healthy
- **Corn:** Common Rust, Northern Leaf Blight, Gray Leaf Spot, Healthy
- **Grape:** Black Rot, Esca, Leaf Blight, Healthy
- **Pepper:** Bacterial Spot, Healthy
- **And 10+ more plant species!**

## 🛠️ Technical Details
- **Framework:** PyTorch 2.0
- **Model:** EfficientNet-B0 (Transfer Learning)
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces

## 🎯 How to Use
1. Upload a clear image of a plant leaf
2. Click 'Analyze Plant Health'
3. Get instant diagnosis with confidence percentage

## 📁 Project Structure
```
plant-disease-detector/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── plant_disease_efficientnet_b0.pth  # Trained model
└── class_mapping.json    # Class labels
```

Built with ❤️ by Adnan - Future FAANG AI Engineer 🚀
