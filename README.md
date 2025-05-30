# 🌿 Plant Disease Detection using MobileNetV2

This project detects plant leaf diseases using a deep learning model based on **MobileNetV2** and **Transfer Learning** in TensorFlow/Keras. The model is trained on a dataset of 10 plant disease classes and fine-tuned to improve performance.

---

## 📊 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224x224x3
- **Layers Added**:
  - `Flatten`
  - `Dense(512, relu)`
  - `Dropout(0.5)`
  - `Dense(10, softmax)`
- **Callbacks**:
  - EarlyStopping
  - ReduceLROnPlateau

---

## 🧠 Trained Model

The trained model file (`final_model_fine_tuned.keras`) is too large for GitHub, but you can download it here:

👉 [Download from Google Drive](https://drive.google.com/file/d/1-3sZocSsMg3xuv6N_OFiZRPXh03am-Sn/view?usp=sharing)

> Replace this link with your actual shareable Drive link.

---

## 📁 Dataset Structure

Dataset should be organized like this for `ImageDataGenerator`:


- Images are resized to `224x224`
- `class_mode = 'categorical'`

---

## 🚀 Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/vvsubhash2615/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
