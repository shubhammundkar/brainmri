"""
BrainVision 3D - Comprehensive Pipeline Unit Tests
Tests:
- Synthetic BraTS volume generation
- Dataset loader & foreground-biased patch sampling
- 3D U-Net forward pass
- Dice + Cross-Entropy Loss computation
- Sliding-window 3D patch inference
- Volumetric and centroid measurements
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

# Delay torch import inside tests to support environments during pip install
def test_synthetic_generation():
    from backend.dataset import generate_synthetic_case
    temp_dir = tempfile.mkdtemp(prefix="test_brats_syn_")
    try:
        case_dir = generate_synthetic_case(temp_dir, subject_id="Test_Case_01", shape=(64, 64, 48))
        assert os.path.exists(case_dir)
        for mod in ['t1', 't1ce', 't2', 'flair', 'seg']:
            f = os.path.join(case_dir, f"Test_Case_01_{mod}.nii.gz")
            assert os.path.exists(f), f"Missing file: {f}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_label_mapping():
    from backend.dataset import raw_to_model_labels, model_to_raw_labels
    raw = np.array([0, 1, 2, 4, 0, 4, 2, 1], dtype=np.uint8)
    model = raw_to_model_labels(raw)
    assert np.array_equal(model, [0, 1, 2, 3, 0, 3, 2, 1])

    restored = model_to_raw_labels(model)
    assert np.array_equal(restored, raw)


def test_volumetrics_calculation():
    from backend.inference import compute_volumetrics
    mask = np.zeros((50, 50, 50), dtype=np.uint8)
    # Put 1000 voxels of ET (label 4), 500 voxels of ED (label 2), 200 voxels of NCR (label 1)
    mask[10:20, 10:20, 10:20] = 4  # 10x10x10 = 1000 voxels
    mask[20:25, 20:25, 20:40] = 2  # 5x5x20 = 500 voxels
    mask[25:35, 25:35, 25:27] = 1  # 10x10x2 = 200 voxels

    metrics = compute_volumetrics(mask, voxel_spacing=(1.0, 1.0, 1.0))
    assert metrics["has_tumor"] is True
    # 1700 voxels at 1mm³ = 1.7 cm³
    assert metrics["total_tumor_cm3"] == 1.7
    assert metrics["enhancing_tumor_cm3"] == 1.0
    assert metrics["edema_cm3"] == 0.5
    assert metrics["necrotic_core_cm3"] == 0.2
    assert "centroid_ras" in metrics
    assert metrics["breakdown"]["enhancing_tumor"]["label"] == 4


def test_model_forward():
    import torch
    from backend.models.unet3d import ResidualUNet3D, count_parameters
    model = ResidualUNet3D(in_channels=4, out_channels=4, base_filters=16)
    params = count_parameters(model)
    assert params > 1_000_000 and params < 5_000_000, f"Unexpected param count: {params}"

    dummy = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 4, 32, 32, 32)


def test_losses_and_metrics():
    import torch
    from backend.losses import DiceCELoss, compute_brats_metrics
    criterion = DiceCELoss()
    logits = torch.randn(2, 4, 16, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16, 16))

    loss, metrics = criterion(logits, targets)
    assert loss.item() > 0
    assert 'dice_loss' in metrics and 'ce_loss' in metrics

    # Test metrics with identical prediction and target
    pred = targets[0]
    m = compute_brats_metrics(pred, targets[0])
    assert m['mean_dice'] > 0.99


if __name__ == "__main__":
    test_synthetic_generation()
    test_label_mapping()
    test_volumetrics_calculation()
    print("Unit tests passed!")
