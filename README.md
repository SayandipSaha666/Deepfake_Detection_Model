# 🎭 DeepFake Detection using Multi-CNN Feature Fusion & Transformer-Based Temporal Modeling

This project presents a state-of-the-art pipeline for **DeepFake video detection** by combining multiple CNN backbones for spatial feature extraction and a **Transformer encoder** for temporal modeling across video frames. The model captures both **frame-level artifacts** and **temporal inconsistencies**, achieving exceptional accuracy.

[📂 Dataset (Kaggle)](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)  
[🧠 Model Notebook](https://www.kaggle.com/code/noobcoder27/deepfake-detection-model/edit)

---

## 🔍 What are DeepFakes?

DeepFakes are synthetic media where a person in a video is digitally altered to say or do things they never actually did, typically using deep learning methods such as GANs or autoencoders. They pose major security, misinformation, and ethical challenges.

---

## 📌 Highlights

- 🔥 100% Accuracy, F1 Score, and ROC-AUC on test data
- 📦 Uses 1000 real/fake video clips from the DFD dataset
- 🧠 Extracts deep spatial features with CNNs (ResNet-50, VGG-19, Xception)
- ⏱️ Captures temporal artifacts with a Transformer encoder
- 🎯 Robust against flickering, blending artifacts, and unnatural transitions

---

## 🧱 Architecture

```text
           +-------------+
           |  Video Clip |
           +------|------+
                  ▼
        +--------------------+
        | Face/Frame Extractor|
        +--------|-----------+
                 ▼
      +---------------------------+
      |  CNN Feature Extractors   |   ← ResNet-50, VGG-19, Xception
      +---+----------+-----------+
          ▼          ▼
      Frame-Level Feature Vectors
                 ↓
     +-----------------------------+
     | Attention-Based Feature Fusion |
     +-----------------------------+
                 ↓
     +-----------------------------+
     | Transformer Encoder (Temporal)|
     +-----------------------------+
                 ↓
     [CLS] Token Representation
                 ↓
        Video-Level Classifier
                 ↓
            Real / Fake


