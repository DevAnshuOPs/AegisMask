"""
model.py - MobileNetV2 Transfer Learning Architecture for Mask Compliance Classification
Features:
- Pretrained MobileNetV2 backbone
- Clean single-dropout replacement head (avoids redundant double-dropout trap)
- Utility methods for progressive unfreezing of top inverted residual blocks
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MaskClassifier(nn.Module):
    """
    MobileNetV2-based 3-class face mask compliance classifier.
    Classes:
        0: with_mask
        1: without_mask
        2: mask_weared_incorrect
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True):
        super(MaskClassifier, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        # Freeze all backbone feature layers initially if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Clean single-dropout replacement of model.classifier
        # Inverted residual last_channel = 1280
        in_features = self.backbone.last_channel
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def freeze_backbone(self):
        """Freezes all layers in the feature extractor backbone."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all layers in the feature extractor backbone."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks: int = 4):
        """
        Unfreezes the last `num_blocks` inverted residual blocks of MobileNetV2.
        model.features contains 19 elements (0=conv2d, 1..17=inverted residual blocks, 18=1x1 conv).
        Unfreezing the top 4 blocks allows high-level feature adaptation without losing early edge/texture detectors.
        """
        for block in self.backbone.features[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def get_trainable_params_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True) -> MaskClassifier:
    """Factory helper function to build and initialize MaskClassifier."""
    return MaskClassifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    print("=" * 60)
    print(" [MaskClassifier Architecture Verification]")
    print("=" * 60)
    
    model = build_model(num_classes=3, pretrained=True, freeze_backbone=True)
    counts_frozen = model.get_trainable_params_count()
    print(f"Stage 1 (Frozen Backbone):")
    print(f"  Total params:     {counts_frozen['total']:,}")
    print(f"  Trainable params: {counts_frozen['trainable']:,} (Classifier head only)")
    print(f"  Frozen params:    {counts_frozen['frozen']:,}")

    model.unfreeze_top_blocks(num_blocks=4)
    counts_unfrozen = model.get_trainable_params_count()
    print(f"\nStage 2 (Top 4 Blocks Unfrozen):")
    print(f"  Total params:     {counts_unfrozen['total']:,}")
    print(f"  Trainable params: {counts_unfrozen['trainable']:,}")
    print(f"  Frozen params:    {counts_unfrozen['frozen']:,}")

    # Test forward pass with dummy batch
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"\nDummy forward pass output shape: {out.shape} -> Expected [2, 3]")
    assert out.shape == (2, 3), f"Unexpected shape {out.shape}"
    print("[✓] Model architecture verified successfully.")
    print("=" * 60)
