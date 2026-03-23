# 🌸 Flower Classification & Color Analysis AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

A dual-pipeline Computer Vision web application that identifies flower species using Deep Learning and mathematically extracts dominant color profiles using Unsupervised Machine Learning. 

## 📖 Project Overview

This project was developed as a comprehensive Computer Vision system capable of robust image analysis. Rather than simply classifying an image, the application runs a simultaneous two-step pipeline to provide rich, diagnostic feedback to the user.

### Core Features
* **Species Classification:** Utilizes a custom-trained **MobileNetV2** architecture via Transfer Learning to classify flower species with high accuracy.
* **Strict Confidence Thresholding:** Incorporates an 80% probability threshold to actively reject invalid uploads (e.g., non-flower images).
* **Dominant Color Extraction:** Employs **K-Means Clustering** on center-cropped regions of interest to extract the true RGB color of the flower petals, ignoring background foliage.
* **Algorithmic Color Matching:** Uses `scipy.spatial.KDTree` to map raw RGB arrays to the closest human-readable color names based on Euclidean distance.
* **Interactive UI:** A real-time, user-friendly dashboard built with **Streamlit**, featuring progress bars, color swatches, and RGB intensity histograms.

---

## 🏗️ Architecture & Technologies

1. **Deep Learning Model:** `MobileNetV2` (Pre-trained on ImageNet). Chosen for its use of depthwise separable convolutions, allowing it to run efficiently in real-time without requiring a dedicated GPU.
2. **Computer Vision:** `OpenCV` for image decoding, dynamic cropping, and RGB histogram generation.
3. **Machine Learning:** `scikit-learn` (K-Means) for pixel-level color clustering.
4. **Web Framework:** `Streamlit` for rendering the front-end user interface.

---

## 🚀 How to Run Locally

*Note: The dataset and trained `.h5` model files are not included in this repository due to GitHub size constraints. You will need to train the model locally before running the web app.*

### 1. Clone the Repository
```bash
git clone [https://github.com/ritika913/Flower-Classifier.git](https://github.com/ritika913/Flower-Classifier.git)
cd Flower-Classifier
