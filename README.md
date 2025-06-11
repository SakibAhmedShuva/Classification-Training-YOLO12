# 🎯 YOLOv12 Classification Training Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00a693?style=for-the-badge&logo=opensourceinitiative&logoColor=white)
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-f9ab00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-blue?style=for-the-badge)

**A complete, end-to-end pipeline for training YOLOv12-cls models on custom binary image classification tasks**

*Optimized for Google Colab with GPU support • Resumable training • Production-ready*

</div>

---

## 🌟 Overview

This repository provides a comprehensive solution for training state-of-the-art YOLOv12 classification models. Built with accessibility in mind, the entire workflow is contained within a single Jupyter notebook that runs seamlessly on Google Colab's free GPU resources.

### 🚀 Key Highlights

- **🔥 State-of-the-Art**: Leverages YOLOv12-cls architecture for superior classification performance
- **🔄 Resumable Training**: Robust checkpoint system protects against interruptions
- **☁️ Cloud-Ready**: Optimized for Google Colab with automatic Drive integration
- **📊 Complete Pipeline**: From data preparation to model validation
- **📖 Well-Documented**: Clear instructions and comprehensive comments

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🛠 Prerequisites](#-prerequisites)
- [🚀 Quick Start](#-quick-start)
- [📖 Detailed Usage Guide](#-detailed-usage-guide)
- [🧠 Model Architecture](#-model-architecture)
- [🖼 Dataset Information](#-dataset-information)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 **Model Performance**
- State-of-the-art YOLOv12-cls architecture
- 34.1M parameters with 126.5 GFLOPs
- Optimized for binary classification tasks

### 💻 **Easy to Use**
- Single notebook workflow
- One-click Colab deployment
- Automatic environment setup

</td>
<td width="50%">

### 🔧 **Robust Training**
- Resumable training sessions
- Automatic checkpoint management
- Google Drive integration for persistence

### 📊 **Comprehensive**
- Complete data pipeline
- Training, validation, and testing
- Performance monitoring and logging

</td>
</tr>
</table>

---

## 📁 Project Structure

```
YOLOv12-Classification/
│
├── 📓 yolo12x-Multi-GPU.ipynb      # Main training notebook
├── 📄 README.md                    # Project documentation
└── 📜 LICENSE                      # MIT License
```

---

## 🛠 Prerequisites

<div align="center">

| Requirement | Description |
|-------------|-------------|
| 🔗 **Google Account** | Access to Google Colab and Drive |
| 🖥️ **GPU Runtime** | T4 GPU recommended (free tier available) |
| 📦 **Dependencies** | Auto-installed via notebook |

</div>

### 📚 Auto-Installed Libraries

- `ultralytics` - YOLOv12 framework
- `torch` & `torchvision` - PyTorch ecosystem
- `ipywidgets` - Interactive widgets
- `gdown` - Google Drive downloads

---

## 🚀 Quick Start

<div align="center">

### 1️⃣ Click the Colab badge above
### 2️⃣ Enable GPU runtime
### 3️⃣ Run all cells
### 4️⃣ Start training!

</div>

---

## 📖 Detailed Usage Guide

### 1️⃣ **Environment Setup**

```python
# Install required packages
!pip install -U ultralytics ipywidgets torchvision

# Import libraries and configure environment
from ultralytics import YOLO
import torch
import os

# Disable Weights & Biases for cleaner output
os.environ['WANDB_MODE'] = 'disabled'
```

### 2️⃣ **Dataset Download & Preparation**

```bash
# Download dataset from Google Drive
!gdown 1YE48CpAz

# Extract to working directory
!unzip "/content/dataset_split.zip" -d /content/
```

> 📍 **Dataset Location**: `/content/dataset_split`

### 3️⃣ **Workspace Configuration**

```python
from google.colab import drive

# Mount Google Drive for persistent storage
drive.mount('/content/drive')

# Create project directory
target_dir = "/content/drive/MyDrive/DS/New-Car"
os.makedirs(target_dir, exist_ok=True)
%cd "{target_dir}"
```

### 4️⃣ **Training Commands**

<details>
<summary><b>🆕 Start New Training</b></summary>

```bash
DATA_DIR="/content/dataset_split"

!yolo task=classify \
      mode=train \
      model=yolo12x-cls.yaml \
      data='{DATA_DIR}' \
      epochs=1500 \
      device=0 \
      batch=16 \
      workers=8 \
      patience=300 \
      seed=101
```
</details>

<details>
<summary><b>🔄 Resume Training</b></summary>

```bash
DATA_DIR="/content/dataset_split"

!yolo task=classify \
      mode=train \
      model=./runs/classify/train/weights/last.pt \
      data='{DATA_DIR}' \
      epochs=1500 \
      device=0 \
      batch=16 \
      workers=16 \
      seed=101 \
      patience=300 \
      resume=True
```
</details>

<details>
<summary><b>✅ Model Validation</b></summary>

```bash
DATA_DIR="/content/dataset_split"

!yolo task=classify \
      mode=val \
      model=./runs/classify/train/weights/best.pt \
      data='{DATA_DIR}'
```
</details>

---

## 🧠 Model Architecture

<div align="center">

### YOLOv12x-cls Specifications

| Metric | Value |
|--------|-------|
| **Architecture** | YOLOv12x-cls |
| **Layers** | 312 |
| **Parameters** | 34.1M |
| **GFLOPs** | 126.5 |
| **Task** | Binary Classification |

</div>

---

## 🖼 Dataset Information

### 📊 Dataset Structure

```
dataset_split/
├── 📁 train/           # Training data
│   ├── 📁 correct/     # Positive class images
│   └── 📁 incorrect/   # Negative class images
├── 📁 val/             # Validation data
│   ├── 📁 correct/
│   └── 📁 incorrect/
└── 📁 test/            # Test data
    ├── 📁 correct/
    └── 📁 incorrect/
```

### 📈 Dataset Statistics

<div align="center">

| Split | Images | Classes |
|-------|--------|---------|
| **Training** | 2,670 | 2 |
| **Validation** | 890 | 2 |
| **Testing** | 892 | 2 |
| **Total** | **4,452** | **2** |

</div>

---

## 📈 Results

### 🎯 Performance Metrics

<div align="center">

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | 76.7% *(after 79 epochs)* |
| **Model Size** | 34.1M parameters |
| **Training Time** | Variable (depends on epochs) |

</div>

> 💡 **Note**: Results shown are intermediate metrics. Final performance will be available after complete training run.

### 📊 Training Progress

The notebook provides real-time feedback including:
- ✅ Loss curves
- 📈 Accuracy metrics  
- 🔄 Learning rate schedules
- ⏱️ Training time estimates

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

<div align="center">

[![Issues](https://img.shields.io/badge/Report-Issues-red?style=for-the-badge&logo=github)](https://github.com/your-repo/issues)
[![Pull Requests](https://img.shields.io/badge/Submit-Pull%20Requests-green?style=for-the-badge&logo=github)](https://github.com/your-repo/pulls)
[![Discussions](https://img.shields.io/badge/Join-Discussions-blue?style=for-the-badge&logo=github)](https://github.com/your-repo/discussions)

</div>

### 🛠 Types of Contributions

- 🐛 Bug reports and fixes
- ✨ Feature requests and implementations  
- 📖 Documentation improvements
- 🧪 Testing and validation
- 💡 Ideas and suggestions

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*See the [LICENSE](LICENSE) file for full details*

</div>

---

## 🙏 Acknowledgements

<div align="center">

### Special Thanks To

[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-blue?style=for-the-badge&logo=python)](https://ultralytics.com/)
*For the powerful and intuitive YOLO framework*

[![Google Colab](https://img.shields.io/badge/Google-Colab-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)
*For providing accessible GPU computing resources*

[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
*For the robust deep learning foundation*

</div>

---

<div align="center">

### 🌟 Star this repository if it helped you!

[![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)](https://github.com/your-username/your-repo)

**Made with ❤️ for the Computer Vision Community**

</div>
