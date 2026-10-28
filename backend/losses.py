"""
BrainVision 3D - Loss Functions and Evaluation Metrics for BraTS Tumor Segmentation

Includes:
- Soft Dice Loss & Multi-Class Generalized Dice Loss
- Combined Dice + Cross-Entropy Loss (DiceCELoss)
- Official BraTS evaluation metrics:
  - Whole Tumor (WT = 1 + 2 + 4)
  - Tumor Core (TC = 1 + 4)
  - Enhancing Tumor (ET = 4)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss with smooth factor.
    Computes average Dice loss across foreground classes (ignoring background by default).
    """
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: shape (B, C, D, H, W)
        targets: shape (B, D, H, W) with integer class values in [0, C-1]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, D, H, W)
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        start_class = 0 if self.include_background else 1
        dice_losses = []

        for c in range(start_class, num_classes):
            p = probs[:, c]
            y = targets_one_hot[:, c]

            intersection = torch.sum(p * y)
            cardinality = torch.sum(p * p) + torch.sum(y * y)

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_losses.append(1.0 - dice)

        if len(dice_losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(dice_losses).mean()


class DiceCELoss(nn.Module):
    """
    Hybrid loss: Weighted combination of Soft Dice Loss and Cross-Entropy Loss.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: torch.Tensor = None,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = SoftDiceLoss(smooth=smooth, include_background=False)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total_loss = self.ce_weight * ce + self.dice_weight * dice

        metrics = {
            'loss': float(total_loss.item()),
            'ce_loss': float(ce.item()),
            'dice_loss': float(dice.item())
        }
        return total_loss, metrics


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Compute binary Dice score between binary boolean/0-1 tensors."""
    pred_bool = (pred > 0).float()
    target_bool = (target > 0).float()
    intersection = torch.sum(pred_bool * target_bool).item()
    cardinality = torch.sum(pred_bool).item() + torch.sum(target_bool).item()
    if cardinality == 0:
        return 1.0  # True negative perfect score
    return (2.0 * intersection + smooth) / (cardinality + smooth)


def compute_brats_metrics(pred_seg: torch.Tensor, target_seg: torch.Tensor) -> Dict[str, float]:
    """
    Compute official BraTS composite evaluation metrics:
    - Whole Tumor (WT): NCR (1) + ED (2) + ET (3 or 4)
    - Tumor Core (TC): NCR (1) + ET (3 or 4)
    - Enhancing Tumor (ET): ET (3 or 4)

    Works with both raw labels (1, 2, 4) and model labels (1, 2, 3).
    """
    with torch.no_grad():
        # ET check: either 3 (model output) or 4 (raw BraTS)
        pred_et = (pred_seg == 3) | (pred_seg == 4)
        target_et = (target_seg == 3) | (target_seg == 4)

        # TC check: NCR (1) or ET
        pred_tc = (pred_seg == 1) | pred_et
        target_tc = (target_seg == 1) | target_et

        # WT check: any tumor voxel
        pred_wt = (pred_seg > 0)
        target_wt = (target_seg > 0)

        dice_et = compute_dice_score(pred_et, target_et)
        dice_tc = compute_dice_score(pred_tc, target_tc)
        dice_wt = compute_dice_score(pred_wt, target_wt)
        mean_dice = (dice_et + dice_tc + dice_wt) / 3.0

        return {
            'dice_wt': float(dice_wt),
            'dice_tc': float(dice_tc),
            'dice_et': float(dice_et),
            'mean_dice': float(mean_dice)
        }
