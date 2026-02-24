# 🌸 Flower Classification using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/ResNet50-Transfer%20Learning-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-85%25+-brightgreen?style=for-the-badge"/>
</p>

A deep learning project that classifies flower images into **5 categories** using **Transfer Learning with ResNet50**. Built with PyTorch on the [Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle.

---

## 📁 Project Structure

```
flower_classification/
├── train.py               # Training script
├── evaluate.py            # Evaluation & metrics
├── predict.py             # Single image inference + sample grid
├── data/
│   └── flowers/           # Dataset folder (not included — see setup)
│       ├── daisy/
│       ├── dandelion/
│       ├── rose/
│       ├── sunflower/
│       └── tulip/
├── models/
│   ├── best_model.pth     # Saved best model weights (auto-generated)
│   └── class_names.json   # Class names (auto-generated after training)
├── outputs/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
└── README.md
```

---

## 🗂️ Dataset

| Property     | Detail                                      |
|--------------|---------------------------------------------|
| Source       | [Kaggle — Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) |
| Total Images | 4,317                                       |
| Classes      | daisy, dandelion, rose, sunflower, tulip    |
| Split        | 70% Train / 15% Val / 15% Test              |

---

## 🧠 Model Architecture

| Component      | Detail                          |
|----------------|---------------------------------|
| Backbone       | ResNet50 (ImageNet pretrained)  |
| Frozen Layers  | layer1, layer2, layer3          |
| Fine-tuned     | layer4 + FC head                |
| Output         | 5 classes (softmax)             |
| Loss           | CrossEntropyLoss                |
| Optimizer      | Adam (lr = 1e-4)                |
| Scheduler      | StepLR (step=5, gamma=0.5)      |
| Epochs         | 15                              |

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow kaggle
```

### 2. Download the dataset

**Step 1** — Get your Kaggle API key:
Go to [kaggle.com](https://kaggle.com) → Profile → Account → **Create New Token** → downloads `kaggle.json`

**Step 2** — Place it in:

| OS | Location |
|----|----------|
| Windows | `C:\Users\<YourName>\.kaggle\kaggle.json` |
| Mac/Linux | `~/.kaggle/kaggle.json` |

**Step 3** — Download and unzip:

```bash
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip -d data/flowers
```

---

## 🚀 Usage

### Train
```bash
python train.py
```
Trains for 15 epochs and saves the best model to `models/best_model.pth`

### Evaluate
```bash
python evaluate.py
```
Prints accuracy, precision, recall, F1-score and saves confusion matrix

### Predict — single image
```bash
# Simple path
python predict.py --image data/flowers/daisy/image.jpg

# Path with spaces — use quotes
python predict.py --image "C:\My Folder\daisy\image.jpg"
```

### Predict — sample grid
```bash
python predict.py --samples
```
Generates a 12-image grid saved to `outputs/sample_predictions.png`

---

## 📊 Results

### Training Curves
![Training Curves](outputs/training_curves.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### Sample Predictions
![Sample Predictions](outputs/sample_predictions.png)

---

## 🔁 Pipeline Overview

```
Raw Images
    ↓
Data Augmentation (crop, flip, rotate, color jitter)
    ↓
ResNet50 Backbone (pretrained, layer1-3 frozen)
    ↓
Fine-tune layer4 + new FC head (5 classes)
    ↓
CrossEntropyLoss + Adam + StepLR scheduler
    ↓
Best model saved → Evaluate → Predict
```

---

## 📦 Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
pillow
kaggle
```

---

## ⚠️ Windows Note

All scripts use `num_workers=0` and `if __name__ == "__main__":` guards to prevent the Windows multiprocessing crash. Always wrap image paths containing spaces in **quotes**.

---

## 👤 Author

**Ahmed Elmoh**  
GitHub: [@Ahmedelmoh](https://github.com/Ahmedelmoh)