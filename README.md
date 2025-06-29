# ✍️ MNIST Digit Recognition System

![Project Banner](sample_predictions.png)  
*An end-to-end deep learning solution for handwritten digit recognition*

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### 🧠 Deep Learning Models
| Model Type | Accuracy | Training Time | Use Case |
|------------|----------|---------------|----------|
| **Simple CNN** | 98.2% | ~12 min (CPU) | Rapid prototyping |
| **Advanced CNN** | 99.3% | ~45 min (CPU) | Production deployment |

### 🖥️ Web Application
- 🎨 Interactive drawing canvas
- ⚡ Real-time predictions (<500ms)
- 🎯 Challenge game mode
- 📊 Prediction history tracking
- 🎚️ Adjustable brush settings

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/idsaturn07/mnist-digit-recognition.git
cd mnist-digit-recognition

# Install dependencies
pip install -r requirements.txt


## 🚀 Usage

### 1. Model Training

```bash
python src/train.py

## 🚀 Launch Web App

To start the interactive web application, run:

```bash
streamlit run ui/app.py
