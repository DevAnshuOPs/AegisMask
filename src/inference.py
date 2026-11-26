"""
inference.py - Real-Time Live Surveillance Pipeline with Clamped Face ROI Extraction & Batched PyTorch Inference
Features:
- OpenCV Caffe SSD Face Detector with CLAHE contrast enhancement on LAB L-channel (detects dark/black masks).
- Confidence threshold set to 0.45 and 50px minimum crop guard to eliminate background false detections.
- 8-frame rolling probability smoother (deque(maxlen=8)) per face with adaptive centroid matching (prevents flicker).
- Recalibrated decision thresholding: with_mask (Green) when P(0) >= 0.42 and P(0) >= P(1); else mask_weared_incorrect (Amber) when P(2) >= 0.18 or P(1) < 0.68; else without_mask (Red).
- Stage 2 crops extracted directly from CLAHE-enhanced frame with 10% context padding for high contrast on black masks.
- Robust Two-Pass Coordinate Clamping with minimum size guard (Fix #3: eliminates (!ssize.empty()) crashes).
- Dynamic Batching: All face crops in a frame are batched into a single 4D tensor for one forward pass.
- Color-coded bounding boxes: Green (with_mask), Red (without_mask), Amber (mask_weared_incorrect).
- Real-time HUD overlay: Live FPS counter and compliance count analytics badge.
- Clean window management: Responds to 'q', Esc (27), and window close button (cv2.WND_PROP_VISIBLE < 1).
"""

import os
import sys
import time
import math
import argparse
from collections import deque
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from model import build_model, MaskClassifier
    from dataset import CLASS_NAMES, NUM_CLASSES
except ImportError:
    from src.model import build_model, MaskClassifier
    from src.dataset import CLASS_NAMES, NUM_CLASSES

# Color Palette (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 220, 0),     # with_mask: Green
    1: (0, 0, 240),     # without_mask: Red
    2: (0, 190, 255)    # mask_weared_incorrect: Amber / Yellow
}

CLASS_DISPLAY_NAMES = {
    0: "Compliant (Mask OK)",
    1: "Non-Compliant (No Mask)",
    2: "Incorrect Mask"
}


class TrackedFace:
    """
    Maintains a rolling temporal window of classification probabilities for a face.
    Smooths out inter-frame classification jitter/flicker.
    """
    def __init__(self, track_id: int, centroid: Tuple[int, int], probs: np.ndarray):
        self.track_id = track_id
        self.centroid = centroid
        self.history = deque(maxlen=8)
        self.history.append(probs)
        self.disappeared = 0

    def update(self, centroid: Tuple[int, int], probs: np.ndarray):
        self.centroid = centroid
        self.history.append(probs)
        self.disappeared = 0

    @property
    def smoothed_probs(self) -> np.ndarray:
        return np.mean(self.history, axis=0)


class CentroidTracker:
    """
    Associates detected faces across frames based on Euclidean centroid proximity
    with an adaptive match radius max(base_distance, 0.6 * box_width).
    """
    def __init__(self, base_distance: float = 90.0, max_disappeared: int = 15):
        self.next_track_id = 0
        self.tracks: Dict[int, TrackedFace] = {}
        self.base_distance = base_distance
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], np.ndarray]]) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Takes current frame detections: [((x1, y1, x2, y2), raw_probs), ...]
        Returns: [((x1, y1, x2, y2), smoothed_probs), ...]
        """
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            to_remove = []
            for track_id, track in self.tracks.items():
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.tracks[track_id]
            return []

        # Calculate centroids of incoming detections
        input_centroids = []
        for (x1, y1, x2, y2), _ in detections:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))

        # If no active tracks, register all detections
        if len(self.tracks) == 0:
            results = []
            for i, ((box), raw_probs) in enumerate(detections):
                track = TrackedFace(self.next_track_id, input_centroids[i], raw_probs)
                self.tracks[self.next_track_id] = track
                results.append((box, track.smoothed_probs))
                self.next_track_id += 1
            return results

        # Match existing tracks to closest detected centroids
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Compute pairwise distance matrix
        D = np.zeros((len(track_ids), len(input_centroids)), dtype=np.float32)
        for i, (tcx, tcy) in enumerate(track_centroids):
            for j, (icx, icy) in enumerate(input_centroids):
                D[i, j] = math.hypot(tcx - icx, tcy - icy)

        # Greedily match smallest distance pairs
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        results = [None] * len(detections)

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            box, raw_probs = detections[col]
            box_width = float(box[2] - box[0])
            adaptive_radius = max(self.base_distance, 0.6 * box_width)
            if D[row, col] > adaptive_radius:
                continue

            track_id = track_ids[row]
            self.tracks[track_id].update(input_centroids[col], raw_probs)
            results[col] = (box, self.tracks[track_id].smoothed_probs)

            used_rows.add(row)
            used_cols.add(col)

        # Register new tracks for unmatched detections
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            box, raw_probs = detections[col]
            track = TrackedFace(self.next_track_id, input_centroids[col], raw_probs)
            self.tracks[self.next_track_id] = track
            results[col] = (box, track.smoothed_probs)
            self.next_track_id += 1

        # Mark disappeared tracks
        unused_rows = set(range(len(track_ids))) - used_rows
        to_remove = []
        for row in unused_rows:
            track_id = track_ids[row]
            self.tracks[track_id].disappeared += 1
            if self.tracks[track_id].disappeared > self.max_disappeared:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

        return results


def load_face_detector(detector_dir: str):
    """Loads OpenCV Caffe SSD Face Detector."""
    prototxt = os.path.join(detector_dir, "deploy.prototxt")
    caffemodel = os.path.join(detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        raise FileNotFoundError(
            f"Face detector files not found in {detector_dir}.\n"
            f"Please run 'python download_models.py' first."
        )

    print(f"[SSD] Loading Caffe face detector from: {detector_dir}...")
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_classifier(weights_path: str, device: torch.device) -> MaskClassifier:
    """Loads the trained PyTorch MobileNetV2 compliance classifier."""
    model = build_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    
    if os.path.exists(weights_path):
        print(f"[CLASSIFIER] Loading trained weights from: {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  [✓] Loaded checkpoint (Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"[WARN] No trained weights found at {weights_path}.")
        print("  Initializing with pretrained ImageNet weights for testing pipeline visuals...")
        model = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False)

    model = model.to(device)
    model.eval()
    return model


def apply_clahe_contrast_enhancement(frame: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE on the L-channel in LAB color space.
    Enhances contrast for dark/black masks and low-light facial contours.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_face_crop(face_crop: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """
    Converts a BGR face crop into an ImageNet-standardized PyTorch tensor [3, H, W].
    """
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def draw_hud(frame: np.ndarray, fps: float, counts: dict):
    """
    Draws a translucent status dashboard badge at the top of the video frame.
    """
    h, w = frame.shape[:2]
    hud_h = 42
    
    # Create translucent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Compliance statistics
    total = counts.get("total", 0)
    c_ok = counts.get(0, 0)
    c_no = counts.get(1, 0)
    c_inc = counts.get(2, 0)
    compliance_rate = (c_ok / total * 100.0) if total > 0 else 100.0

    hud_text = (
        f"FPS: {fps:4.1f} | "
        f"Faces: {total} | "
        f"Compliant: {c_ok} | "
        f"No Mask: {c_no} | "
        f"Incorrect: {c_inc} | "
        f"Compliance: {compliance_rate:5.1f}%"
    )
    cv2.putText(frame, hud_text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def run_surveillance(
    source=0,
    confidence_thresh: float = 0.45,
    models_dir: str = "models",
    detector_dir: str = "models/face_detector",
    device_name: str = None
):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SURVEILLANCE] Initializing on device: {device}")

    # 1. Load Stage 1 Face Detector
    ssd_net = load_face_detector(detector_dir)

    # 2. Load Stage 2 PyTorch Classifier
    weights_path = os.path.join(models_dir, "mask_classifier.pth")
    classifier = load_classifier(weights_path, device)

    # 3. Initialize Centroid Tracker & Probability Smoother
    tracker = CentroidTracker(base_distance=90.0, max_disappeared=15)

    # 4. Open Video Stream
    print(f"[VIDEO] Connecting to video source: {source}...")
    try:
        source_idx = int(source)
    except ValueError:
        source_idx = source  # Video file path

    cap = cv2.VideoCapture(source_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {source}")
        return

    # Frame window dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    window_name = "Mask Compliance Surveillance - Live Feed"
    print(f"[VIDEO] Stream started: {frame_w}x{frame_h}. Press 'q' or 'Esc' to exit, 's' to snapshot.")

    fps = 0.0
    prev_time = time.time()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video feed ended or camera disconnected.")
            break

        h, w = frame.shape[:2]

        # Apply CLAHE on LAB L-channel
        # Enhances dark/black masks, fabric textures, and low-contrast facial boundaries
        frame_clahe = apply_clahe_contrast_enhancement(frame)

        # Stage 1: Caffe SSD 300x300 forward pass
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_clahe, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        ssd_net.setInput(blob)
        detections = ssd_net.forward()

        valid_boxes = []
        face_tensors = []

        # Iterate detections [1, 1, N, 7]
        num_detections = detections.shape[2]
        for i in range(num_detections):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence_thresh:
                continue

            # ------------------------------------------------------------------
            # Fix #3: Robust Two-Pass Coordinate Clamping with 50px Minimum Size Guard
            # ------------------------------------------------------------------
            # 1. First clamp normalized coordinates to [0.0, 1.0] and scale to pixel dimensions
            startX = int(max(0.0, min(float(detections[0, 0, i, 3]), 1.0)) * w)
            startY = int(max(0.0, min(float(detections[0, 0, i, 4]), 1.0)) * h)
            endX   = int(max(0.0, min(float(detections[0, 0, i, 5]), 1.0)) * w)
            endY   = int(max(0.0, min(float(detections[0, 0, i, 6]), 1.0)) * h)

            box_w = endX - startX
            box_h = endY - startY
            if box_w < 50 or box_h < 50:
                continue

            # 2. Add 10% context padding (preserves chin/jawline context for incorrect mask wear)
            pad_x = int(0.10 * box_w)
            pad_y = int(0.10 * box_h)

            # 3. Second clamp padded coordinates to frame boundaries
            x1 = max(0, startX - pad_x)
            y1 = max(0, startY - pad_y)
            x2 = min(w, endX + pad_x)
            y2 = min(h, endY + pad_h)

            # 4. Skip small/degenerate crops (< 50px) to eliminate background false detections
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            # Extract Stage 2 classifier crop from frame_clahe so black masks retain fabric contrast
            face_roi = frame_clahe[y1:y2, x1:x2]
            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue

            valid_boxes.append((x1, y1, x2, y2))
            face_tensors.append(preprocess_face_crop(face_roi, img_size=224))

        raw_detections = []

        # Stage 2: Batched PyTorch Inference
        if len(face_tensors) > 0:
            batch_tensor = torch.stack(face_tensors).to(device)
            with torch.no_grad():
                logits = classifier(batch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for box, prob_vec in zip(valid_boxes, probs):
                raw_detections.append((box, prob_vec))

        # Fix (2): 5-frame rolling probability smoother per face via centroid tracking
        smoothed_detections = tracker.update(raw_detections)

        counts = {"total": len(smoothed_detections), 0: 0, 1: 0, 2: 0}

        # Render bounding boxes and prediction tags with recalibrated decision thresholding
        for (x1, y1, x2, y2), smoothed_probs in smoothed_detections:
            # Recalibrated decision thresholds for dark masks & low light:
            # 1. with_mask (class 0, Green) when smoothed_probs[0] >= 0.42 and smoothed_probs[0] >= smoothed_probs[1]
            # 2. Otherwise mask_weared_incorrect (class 2, Amber) when smoothed_probs[2] >= 0.18 or smoothed_probs[1] < 0.68
            # 3. Otherwise without_mask (class 1, Red)
            if smoothed_probs[0] >= 0.42 and smoothed_probs[0] >= smoothed_probs[1]:
                pred_cls = 0
            elif smoothed_probs[2] >= 0.18 or smoothed_probs[1] < 0.68:
                pred_cls = 2
            else:
                pred_cls = 1

            pred_conf = float(smoothed_probs[pred_cls])

            counts[pred_cls] = counts.get(pred_cls, 0) + 1
            color = CLASS_COLORS.get(pred_cls, (255, 255, 255))
            label_text = f"{CLASS_NAMES[pred_cls]}: {pred_conf * 100:.1f}%"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label banner above bounding box
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            banner_y1 = max(0, y1 - th - 8)
            banner_y2 = y1
            cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 6, banner_y2), color, -1)
            cv2.putText(
                frame, label_text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps
        prev_time = curr_time

        # Draw HUD
        draw_hud(frame, fps, counts)

        # Show frame
        cv2.imshow(window_name, frame)

        # Fix (4): Handle keys ('q', Esc=27, 's') and window close button (X)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[INFO] Exit key pressed ('q' or 'Esc'). Terminating surveillance pipeline...")
            break
        elif key == ord('s'):
            snapshot_path = os.path.join(models_dir, f"snapshot_{int(time.time())}.png")
            cv2.imwrite(snapshot_path, frame)
            print(f"[SNAPSHOT] Saved current frame to: {snapshot_path}")

        # Check if window was closed via 'X' button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[INFO] Window closed by user ('X' button). Terminating surveillance pipeline...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream released cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Face Mask Compliance Surveillance Pipeline")
    parser.add_argument("--source", default=0, help="Webcam index (0) or path to video file")
    parser.add_argument("--confidence", type=float, default=0.45, help="SSD detection confidence threshold (default: 0.45)")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model weights")
    parser.add_argument("--detector-dir", type=str, default="models/face_detector", help="Caffe SSD directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_path = os.path.join(project_root, args.models_dir)
    detector_path = os.path.join(project_root, args.detector_dir)

    run_surveillance(
        source=args.source,
        confidence_thresh=args.confidence,
        models_dir=models_path,
        detector_dir=detector_path,
        device_name=args.device
    )
