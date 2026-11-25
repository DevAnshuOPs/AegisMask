# AegisMask — Public Health Biometric Face Mask and Compliance Verification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-94.67%25-brightgreen.svg)]()
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.8360-success.svg)]()

> **AegisMask** is an autonomous, two-stage cascaded Computer Vision and Deep Learning pipeline engineered for real-time respiratory compliance monitoring in crowded public spaces (transit hubs, clinical facilities, airports). Running at **30+ FPS on a standard laptop CPU**, it localizes human faces and accurately categorizes compliance into three operational states:
> 1. 🟢 **Compliant** (`with_mask` — Green)
> 2. 🔴 **Non-Compliant** (`without_mask` — Red)
> 3. 🟡 **Improperly Worn / Non-Compliant** (`mask_weared_incorrect` — Amber: nose exposed or chin-dropped)

---

## 1. System Architecture & Engineering

AegisMask decouples face localization from multi-class compliance classification, combining C++ OpenCV DNN execution with transfer-learned PyTorch inference:

```
[ LIVE CAMERA / VIDEO FEED ]
             |
             v
[ STAGE 1: CONTRAST ENHANCEMENT ] ---> LAB color space CLAHE on L-channel (enhances dark masks & dim lighting)
             |
             v
[ STAGE 1: OPENCV CAFFE SSD ] -------> ResNet-10 Caffe SSD face detection at 300x300 (confidence >= 0.45)
             |
             v
[ ROI EXTRACTION & GUARDING ] -------> Two-pass boundary clamping, 10% context padding, 50px minimum crop guard
             |
             v
[ STAGE 2: DYNAMIC TENSOR BATCH ] ---> Preprocess K face crops into single 4D Tensor [K, 3, 224, 224]
             |
             v
[ STAGE 2: PYTORCH MOBILENETV2 ] ----> Single batched forward pass with transfer-learned 3-class linear head
             |
             v
[ TEMPORAL CENTROID TRACKER ] -------> Rolling 5-frame probability smoothing (deque(maxlen=5)) across tracks
             |
             v
[ CALIBRATED DECISION HUD ] ---------> Green (P0 > 0.48 & max), Amber (P2 >= 0.25 or P1 < 0.55), Red (else)
             |
             v
[ STABILIZED VISUAL OUTPUT ] --------> 30+ FPS live surveillance feed with real-time compliance statistics HUD
```

### Key Technical Pillars
* **Decoupled Architecture**: Decoupling localization (OpenCV Caffe SSD) from classification (PyTorch MobileNetV2) avoids monolithic anchor overhead on full frames, sustaining 30+ FPS on consumer CPUs.
* **LAB-Channel CLAHE**: Applies Contrast Limited Adaptive Histogram Equalization to the luminance channel before face detection, preventing dark/black cloth masks from disappearing into shadow contours.
* **Dynamic Tensor Batching**: Stacks all $K$ detected face crops in a frame into a single 4D tensor `[K, 3, 224, 224]`, executing one batched forward pass rather than $K$ sequential passes.
* **5-Frame Rolling Centroid Tracker**: Associates faces frame-to-frame via Euclidean centroid proximity ($<90\text{ px}$) and averages predictions over a 5-frame rolling window (`deque(maxlen=5)`), completely eliminating bounding-box color flicker.
* **Calibrated Decision Thresholding**: Prevents eyeglasses from triggering false Amber classifications: `with_mask` is locked in Green whenever $P(0) > 0.48$ and dominant, while nose-exposed masks reliably trigger Amber.

---

## 2. Experimental Results & Academic Evaluation

Evaluated on **676 unseen face crops** from the Kaggle Face Mask Detection benchmark (using an **image-level stratified 80/20 split** to guarantee zero background or identity leakage):

| Class Name | Precision | Recall | F1-Score | Support | Visual State |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`with_mask`** (Compliant) | **0.9845** | **0.9566** | **0.9703** | 530 | 🟢 Green BBox |
| **`without_mask`** (Non-Compliant) | **0.8976** | **0.9744** | **0.9344** | 117 | 🔴 Red BBox |
| **`mask_weared_incorrect`** (Improper) | **0.5588** | **0.6552** | **0.6032** | 29 | 🟡 Amber BBox |
| **Overall Accuracy** | — | — | **0.9467** | **676** | **640 / 676 Correct** |
| **Macro Average** | **0.8136** | **0.8620** | **0.8360** | **676** | **Balanced Performance** |
| **Weighted Average** | **0.9512** | **0.9467** | **0.9484** | **676** | — |

### Confusion Matrix

<p align="center">
  <img src="models/confusion_matrix.png" alt="AegisMask Confusion Matrix" width="550"/>
</p>

### Live Surveillance Demonstration

<p align="center">
  <img src="models/snapshot_1795099385.png" alt="AegisMask Live Surveillance Snapshot" width="650"/>
</p>

* **Minority Class Handling**: Despite `mask_weared_incorrect` comprising only 3.18% of training data (93 crops), our smoothed inverse square-root loss weighting ($5.22:1$ ratio vs. harsh $27.2:1$) combined with Epoch 6 progressive unfreezing achieved **65.52% Recall** and **0.6032 F1**, where standard baselines collapse to $<0.20$.

---

## 3. Quick Start (4 Steps)

Run live inference immediately using our pre-trained model checkpoint (`models/mask_classifier.pth` included):

### Step 1: Clone the Repository
```bash
git clone https://github.com/DevAnshuOPs/AegisMask.git
cd AegisMask
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify / Download Face Detector Assets
```bash
python download_models.py
```
*(Verifies `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in `models/face_detector/`)*

### Step 4: Launch Live Webcam Surveillance
```bash
python src/inference.py --source 0
```

#### Interactive Controls:
* **`q`** or **`Esc`** or window **`X`**: Close the surveillance window cleanly.
* **`s`**: Save a high-resolution timestamped snapshot PNG to `models/` for presentation slides.

---

## 4. Retraining & Reproducibility

If you wish to retrain the model from scratch on the Kaggle Face Mask Detection dataset:

1. **Extract Dataset**:
   Download the Kaggle Face Mask Detection dataset and extract:
   - 853 images into `data/images/`
   - 853 XMLs into `data/annotations/`
   *(Or run `python extract_dataset.py` with `archive.zip` in the root folder).*

2. **Verify Dataset Split & Smoothed Weights**:
   ```bash
   python src/dataset.py
   ```

3. **Train with Progressive Fine-Tuning**:
   ```bash
   python src/train.py --batch-size 32 --epochs 20
   ```
   * **Epochs 1–5**: Backbone frozen; linear head warmed up at $\text{lr}=10^{-3}$.
   * **Epochs 6–20**: Unfreezes top 4 inverted residual blocks (`features[-4:]`) with fresh differential Adam ($\text{lr}=10^{-4}$ backbone, $10^{-3}$ head) + Cosine Annealing.
   * Automatically saves best checkpoint to `models/mask_classifier.pth` and exports `models/confusion_matrix.png`.

---

## 5. Repository Structure

```plaintext
AegisMask/
├── models/
│   ├── face_detector/
│   │   ├── deploy.prototxt                    # OpenCV Caffe SSD network definition
│   │   └── res10_300x300_ssd_iter_140000.caffemodel  # Pretrained Caffe SSD weights
│   ├── mask_classifier.pth                    # Best PyTorch MobileNetV2 checkpoint (Epoch 14)
│   ├── confusion_matrix.png                   # High-res 300 DPI evaluation confusion matrix
│   └── snapshot_1795099385.png                # Live webcam surveillance sample
├── src/
│   ├── __init__.py
│   ├── dataset.py                             # Image-level stratified split, VOC XML parser, cropper
│   ├── model.py                               # MobileNetV2 transfer learning & progressive unfreezing
│   ├── train.py                               # Two-stage differential training loop & metrics generator
│   └── inference.py                           # Crash-proof CLAHE SSD + batched PyTorch live engine
├── download_models.py                         # Automated downloader for Caffe SSD detector files
├── extract_dataset.py                         # Archive extraction utility
├── PROJECT_DOCUMENTATION.txt                  # Comprehensive report source & viva defense Q&A bank
├── requirements.txt                           # Core dependencies
├── .gitignore                                 # Clean git tracking exclusions
└── README.md
```

---

## 6. Academic Attribution & Coursework

Developed as part of **CSE471: Computer Vision & Deep Learning**.
* **Primary Reference Document**: Detailed architectural rationales, 11-slide presentation outline, and 10 faculty viva questions & answers are available in [PROJECT_DOCUMENTATION.txt](PROJECT_DOCUMENTATION.txt).

---

## 7. License

This project is open-source under the [MIT License](LICENSE).
