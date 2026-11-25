"""
train.py - Training Pipeline with Progressive Unfreezing & Academic Evaluation
Features:
- Progressive fine-tuning: Epochs 1-5 frozen backbone (lr=1e-3), Epochs 6-20 unfreeze top 4 blocks with fresh differential-LR Adam optimizer (1e-4 backbone, 1e-3 head).
- Smoothed inverse square-root class weights in CrossEntropyLoss.
- Checkpoints best model to models/mask_classifier.pth.
- Generates and saves high-resolution confusion matrix to models/confusion_matrix.png and prints classification report.
"""

import os
import sys
import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib
matplotlib.use('Agg')  # Headless backend for saving figures without display server
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
    from model import build_model, MaskClassifier
except ImportError:
    from src.dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
    from src.model import build_model, MaskClassifier


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False, file=sys.stdout)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, list, list]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False, file=sys.stdout)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return val_loss, val_acc, val_macro_f1, all_labels, all_preds


def save_confusion_matrix_plot(y_true, y_pred, save_path: str, class_names=CLASS_NAMES):
    """
    Renders and saves a normalized confusion matrix plot for viva/project presentation.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    plt.title("Face Mask Compliance - Confusion Matrix", fontsize=13, pad=12, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='semibold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='semibold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[✓] Confusion matrix plot saved to: {save_path}")


def run_training(
    data_dir: str = "data",
    models_dir: str = "models",
    batch_size: int = 32,
    total_epochs: int = 20,
    stage1_epochs: int = 5,
    device_name: str = None
):
    # Setup hardware device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Training on: {device}")

    os.makedirs(models_dir, exist_ok=True)
    best_weights_path = os.path.join(models_dir, "mask_classifier.pth")
    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")

    # 1. Load Data
    print(f"\n[DATA] Loading datasets from: {data_dir}")
    train_loader, val_loader, class_weights, train_counts, val_counts = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, img_size=224, num_workers=0
    )

    print(f"  Training samples:   {len(train_loader.dataset)} face crops (Batch size: {batch_size})")
    print(f"  Validation samples: {len(val_loader.dataset)} face crops")
    print(f"  Smoothed Class Weights: {class_weights.tolist()}")
    for idx, cname in enumerate(CLASS_NAMES):
        print(f"    - {cname:<25}: count={train_counts.get(idx, 0):4d} | weight={class_weights[idx]:.4f}")

    # 2. Build Model with Frozen Backbone (Stage 1)
    print(f"\n[MODEL] Initializing MobileNetV2 with frozen backbone...")
    model = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # 3. Loss with smoothed class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 4. Stage 1 Optimizer: Head only (Epochs 1..stage1_epochs)
    print(f"[STAGE 1] Epochs 1-{stage1_epochs}: Training classifier head only (lr=1e-3)...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_f1 = 0.0

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        # Trigger Stage 2 at epoch (stage1_epochs + 1)
        if epoch == (stage1_epochs + 1):
            print("\n" + "=" * 70)
            print(f"[STAGE 2] Epoch {epoch}: Unfreezing top 4 inverted residual blocks (model.features[-4:])...")
            model.unfreeze_top_blocks(num_blocks=4)
            param_counts = model.get_trainable_params_count()
            print(f"  Trainable parameters increased to: {param_counts['trainable']:,}")
            
            # Instantiate fresh differential-LR Adam optimizer
            print(f"  Instantiating fresh differential-LR Adam optimizer (Backbone: 1e-4, Head: 1e-3)...")
            optimizer = torch.optim.Adam([
                {'params': model.backbone.features[-4:].parameters(), 'lr': 1e-4},
                {'params': model.backbone.classifier.parameters(), 'lr': 1e-3}
            ], weight_decay=1e-4)
            
            remaining_epochs = total_epochs - stage1_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=1e-6
            )
            print("=" * 70)

        # Train and Validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] (lr={current_lr:.1e}) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}%, Macro F1: {val_macro_f1:.4f}")

        # Checkpoint on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_macro_f1': val_macro_f1,
                'class_names': CLASS_NAMES
            }, best_weights_path)
            print(f"  --> [SAVED] Best model checkpoint saved to {best_weights_path} (Val Loss: {val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Macro F1: {best_val_f1:.4f}")

    # 5. Final Evaluation & Academic Artifact Generation
    print("\n" + "=" * 70)
    print(" [Academic Evaluation & Viva Defense Metrics]")
    print("=" * 70)
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']}...")

    _, final_acc, final_f1, y_true, y_pred = validate(model, val_loader, criterion, device)
    
    print("\nClassification Report (Per-Class Precision, Recall, F1):")
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)

    # Save Confusion Matrix plot
    save_confusion_matrix_plot(y_true, y_pred, save_path=cm_plot_path, class_names=CLASS_NAMES)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face Mask Compliance Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs with frozen backbone")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Automatically resolve relative paths based on repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, args.data_dir)
    models_path = os.path.join(project_root, args.models_dir)

    run_training(
        data_dir=data_path,
        models_dir=models_path,
        batch_size=args.batch_size,
        total_epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        device_name=args.device
    )
