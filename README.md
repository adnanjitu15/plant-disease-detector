# 🌿 Plant Disease Detector

AI-powered plant disease classification system achieving **98.23% accuracy** across 39 disease classes.

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://huggingface.co/spaces/Zaayan/plant-disease-detector-adnan)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)

---

## Overview

This project addresses the agricultural challenge where farmers lose 20-40% of crops annually to plant diseases. The system provides instant diagnosis through leaf image analysis using deep learning.

**[Try the live demo →](https://huggingface.co/spaces/Zaayan/plant-disease-detector-adnan)**

---

## Features

- Real-time prediction (< 2 seconds)
- 98.23% validation accuracy
- 39 disease classifications
- Confidence threshold warnings for uncertain predictions
- Responsive web interface

---

## Technical Stack

- **Model:** EfficientNet-B0 (Transfer Learning)
- **Framework:** PyTorch
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces
- **Training:** Google Colab (Tesla T4 GPU)

---

## Performance

```
Validation Accuracy:  98.23%
Training Time:        17.5 minutes
Dataset:              PlantVillage (55,448 images)
Classes:              39 plant diseases
```

### Supported Crops

Tomato • Corn • Apple • Grape • Potato • Pepper • Peach • Cherry • Strawberry • Raspberry • Blueberry • Orange

---

## Installation

```bash
# Clone repository
git clone https://github.com/adnanjitu15/plant-disease-detector.git
cd plant-disease-detector

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## Model Training

### Architecture
- Base: EfficientNet-B0 (ImageNet pretrained)
- Output: 39-class softmax classifier
- Input: 224×224 RGB images

### Configuration
```python
Optimizer:     Adam (lr=0.001)
Loss:          CrossEntropyLoss
Batch Size:    32
Epochs:        3
Train/Val/Test: 80/10/10
```

### Data Augmentation
- Random crop and resize
- Horizontal flip
- Rotation (±20°)
- Color jitter

---

## Known Limitations

The model was trained on isolated leaf images with controlled lighting. Performance may degrade for:
- Whole plant photos
- Outdoor lighting variations
- Images with complex backgrounds

**Mitigation:** The application displays confidence warnings when predictions fall below 95% certainty.

---

## Project Structure

```
plant-disease-detector/
├── app.py                              # Streamlit application
├── plant_disease_efficientnet_b0.pth  # Model weights
├── class_mapping.json                  # Class labels
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation
```

---

## Future Work

- Grad-CAM visualization for interpretability
- Multi-disease detection per image
- REST API endpoint
- Mobile application

---

## Author

**Adnan Jitu**  
📧 adnanjitu15@gmail.com  
🔗 [GitHub](https://github.com/adnanjitu15)

---

## Acknowledgments

Built using the PlantVillage dataset. Deployed on Hugging Face Spaces.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
