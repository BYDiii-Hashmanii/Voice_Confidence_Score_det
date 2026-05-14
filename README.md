# 🎙️ Audio Word Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-red?logo=keras&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio-green)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-96.72%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning-based **audio keyword classification system** that identifies spoken words from raw audio using **Mel-Spectrogram** feature extraction and a **Convolutional Neural Network (CNN)**. Trained on 7,469 audio samples across 4 classes, achieving **96.72% validation accuracy**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Inference](#-inference)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Notebooks](#-notebooks)

---

## 🧠 Overview

This project implements an end-to-end audio classification pipeline:

1. **Load** raw `.wav`/`.ogg` audio files using TensorFlow's audio dataset utilities
2. **Extract** Mel-Spectrogram features with Librosa
3. **Train** a 4-layer CNN on the spectrograms
4. **Classify** new audio clips with label and confidence score

The system is capable of recognizing the following spoken keywords:

| Label | Word |
|---|---|
| 0 | `backward` |
| 1 | `house` |
| 2 | `marvin` |
| 3 | `visual` |

---

## 📂 Dataset

| Property | Value |
|---|---|
| Total Samples | 7,469 audio files |
| Training Split | 5,976 (80%) |
| Validation Split | 1,493 (20%) |
| Number of Classes | 4 |
| Sample Rate | 16,000 Hz |
| Sequence Length | 16,000 samples (1 second) |
| Source | Google Drive (`audio_dataset/`) |

The dataset is organized into subdirectories per class (one folder per keyword), which is compatible with `tf.keras.utils.audio_dataset_from_directory`.

```
audio_dataset/
├── backward/
│   └── *.wav
├── house/
│   └── *.wav
├── marvin/
│   └── *.wav
└── visual/
    └── *.wav
```

---

## 🏗️ Model Architecture

The classifier is a **4-block CNN** followed by fully connected layers, designed for image-like Mel-Spectrogram inputs of shape `(110, 110, 3)`.

```
Input: (110, 110, 3)  ← Mel-Spectrogram resized to 110×110, 3 channels
        │
        ▼
┌─────────────────────────┐
│  Conv2D(32, 3×3, ReLU)  │
│  BatchNormalization      │
│  MaxPooling2D(2×2)       │
│  Dropout(0.25)           │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Conv2D(64, 3×3, ReLU)  │
│  BatchNormalization      │
│  MaxPooling2D(2×2)       │
│  Dropout(0.25)           │
└────────────┬────────────┘
             ▼
┌──────────────────────────┐
│  Conv2D(128, 3×3, ReLU)  │
│  BatchNormalization       │
│  MaxPooling2D(2×2)        │
│  Dropout(0.30)            │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Conv2D(256, 3×3, ReLU)  │
│  BatchNormalization       │
│  MaxPooling2D(2×2)        │
│  Dropout(0.30)            │
└────────────┬─────────────┘
             ▼
        Flatten()
             ▼
     Dense(512, ReLU)
   BatchNormalization
       Dropout(0.5)
             ▼
     Dense(4, Softmax)
             │
             ▼
      Output: 4 classes
```

**Optimizer:** Adam  
**Loss:** Categorical Crossentropy  
**Metrics:** Accuracy

---

## 🔄 Training Pipeline

```
Raw Audio (.wav)
      │
      ▼
tf.keras audio_dataset_from_directory
(batch=16, sr=16000, seq_len=16000)
      │
      ▼
librosa.melspectrogram()
  n_mels=110, sr=16000
      │
      ▼
librosa.power_to_db()
      │
      ▼
tf.image.resize → (110, 110, 1)
      │
      ▼
np.repeat → (110, 110, 3)   ← 3-channel for CNN
      │
      ▼
Z-score Normalization
  (x - mean) / (std + 1e-6)
      │
      ▼
to_categorical (one-hot labels)
      │
      ▼
CNN Training (epochs=10, batch=32)
      │
      ▼
Saved Model (.keras)
```

---

## 📊 Results

### Training History

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|---------------|-------------|------------|----------|
| 1 | 71.18% | 35.23% | 0.9589 | 3.9780 |
| 2 | 87.74% | 77.83% | 0.3593 | 0.6723 |
| 3 | 90.53% | 84.73% | 0.2622 | 0.4667 |
| 4 | 91.65% | 84.73% | 0.2420 | 0.5438 |
| 5 | 93.33% | 95.85% | 0.1852 | 0.1143 |
| 6 | 94.72% | 96.32% | 0.1709 | 0.1132 |
| 7 | 95.43% | 96.25% | 0.1350 | 0.1148 |
| 8 | 96.06% | 77.43% | 0.1113 | 0.7320 |
| 9 | 95.41% | 96.45% | 0.1277 | 0.1097 |
| **10** | **96.36%** | **96.72%** | **0.1106** | **0.0925** |

> ✅ **Final Validation Accuracy: 96.72%**

### Inference Example

```
Test File:       visual.ogg
Predicted Label: visual
Confidence:      100.00%
All Probabilities: [4.45e-06, 3.59e-09, 9.08e-07, 9.9999e-01]
```

---

## 🔍 Inference

Once the model is trained and saved, you can classify any audio file:

```python
import librosa
import numpy as np
import tensorflow as tf

label_names = np.array(['backward', 'house', 'marvin', 'visual'])
model = tf.keras.models.load_model('audio_word_classifier.keras')

def extract_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=110)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def preprocess_audio_for_test(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)

    # Pad or trim to 1 second
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))
    else:
        audio = audio[:16000]

    mel = extract_mel(audio)
    mel_resized = tf.image.resize(mel[..., np.newaxis], (110, 110)).numpy()
    mel_3ch = np.repeat(mel_resized, 3, axis=-1)
    mel_3ch = (mel_3ch - mel_3ch.mean()) / (mel_3ch.std() + 1e-6)
    return np.expand_dims(mel_3ch, axis=0)

def classify_audio(audio_path):
    x = preprocess_audio_for_test(audio_path)
    predictions = model.predict(x, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = label_names[predicted_index]
    confidence = predictions[predicted_index]
    return predicted_label, confidence, predictions

# Run inference
label, confidence, probs = classify_audio("your_audio.wav")
print(f"Predicted: {label} | Confidence: {confidence * 100:.2f}%")
```

---

## 📁 Project Structure

```
audio-word-classifier/
│
├── 📓 Audio_Model_train.ipynb       # Full training pipeline
├── 📓 Audio_handling_project.ipynb  # Audio preprocessing & exploration
├── 📓 Confidence_Main.ipynb         # Confidence scoring & analysis
│
├── 📁 audio_dataset/                # Dataset directory (Google Drive)
│   ├── backward/
│   ├── house/
│   ├── marvin/
│   └── visual/
│
├── 🧠 audio_word_classifier.keras   # Saved trained model
└── README.md
```

---

## ⚙️ Requirements

```txt
tensorflow>=2.x
keras
librosa
numpy
```

Install dependencies:

```bash
pip install tensorflow librosa numpy
```

For Google Colab (used during training):

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 Usage

### 1. Train the Model

Open and run `Audio_Model_train.ipynb` in Google Colab:

- Mount Google Drive
- Point dataset directory to `audio_dataset/` in your Drive
- Run all cells — the model trains for 10 epochs and saves to `audio_word_classifier.keras`

### 2. Run Inference

```python
label, confidence, probs = classify_audio("path/to/audio.wav")
print(f"Label: {label}, Confidence: {confidence*100:.2f}%")
```

### 3. Extend the Model

To add new word classes:
- Add a new subdirectory under `audio_dataset/` with audio samples
- Re-run the training notebook
- Update `label_names` in the inference script

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `Audio_Model_train.ipynb` | End-to-end training: data loading → mel-spectrogram → CNN → save model |
| `Audio_handling_project.ipynb` | Audio preprocessing, exploration, and pipeline utilities |
| `Confidence_Main.ipynb` | Confidence scoring, threshold tuning, and result analysis |

---

## 🔬 Key Design Decisions

- **Mel-Spectrogram** chosen over raw waveform for its ability to capture frequency-time patterns efficiently
- **3-channel conversion** (`np.repeat`) enables use of standard image-CNN architectures
- **110×110 resize** balances spectral resolution with memory and compute cost
- **BatchNormalization** at every conv block stabilizes training and accelerates convergence
- **Z-score normalization** applied per-sample at inference to match training distribution

---

## ⚠️ Known Issues & Notes

- **Epoch 8 dip**: A notable drop in val accuracy (77.43%) at epoch 8 suggests occasional batch noise or overfitting spikes — adding `ModelCheckpoint` to save the best weights is recommended
- **Per-sample normalization at inference** differs slightly from batch-level normalization used during training; this is a known tradeoff
- **Dataset scope**: Currently trained on 4 keywords only; generalization to unseen words is not supported

---

## 📄 License

This project is licensed under the **MIT License**.

---

> Built with ❤️ using TensorFlow, Keras, and Librosa on Google Colab
