"""
BrainVision 3D - Patch-Based Sliding Window Inference Engine for BraTS 2020
Target Hardware: NVIDIA RTX 3050 (6 GB VRAM)

Features:
- Sliding window 3D patch inference with 50% overlap and 3D Gaussian blending
- Memory-safe execution strictly within 6 GB VRAM (or CPU fallback)
- Mixed precision (torch.amp / torch.cuda.amp) acceleration
- Brain bounding box optimization
- Volumetric quantitative analysis (cm³) for NCR, ED, ET, WT
- Lesion centroid computation in millimeter (RAS) coordinates
- Export to standard compressed NIfTI (.nii.gz)
"""

import os
import math
from typing import Dict, Tuple, Optional, Union

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from .dataset import normalize_intensity, model_to_raw_labels, extract_brain_bbox
from .models.unet3d import UNet3D


def create_gaussian_weight_window(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """
    Generate a 3D Gaussian weight map for smooth patch overlap blending.
    Center voxels receive higher weights, edges decay smoothly to eliminate seam artifacts.
    """
    pd, ph, pw = patch_size
    d_coords = torch.linspace(-1, 1, pd)
    h_coords = torch.linspace(-1, 1, ph)
    w_coords = torch.linspace(-1, 1, pw)

    d_grid, h_grid, w_grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
    dist_sq = d_grid ** 2 + h_grid ** 2 + w_grid ** 2
    weight = torch.exp(-dist_sq / (2 * (sigma_scale ** 2)))
    # Ensure minimum positive weight at boundaries
    weight = torch.clamp(weight, min=1e-4)
    return weight


class SlidingWindowInferer:
    """
    Sliding window inferer for 3D volumetric segmentation on memory-constrained GPUs.

    Args:
        model: Trained UNet3D PyTorch model
        patch_size: 3D patch dimensions (default: (96, 96, 96))
        overlap: Overlap fraction between adjacent patches (default: 0.5)
        device: 'cuda' or 'cpu'
        use_amp: Whether to use automatic mixed precision (fp16/bf16)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        device: Optional[str] = None,
        use_amp: bool = True
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = model.to(self.device).eval()
        self.patch_size = patch_size
        self.overlap = overlap
        self.use_amp = use_amp and (self.device == 'cuda')

        self.gaussian_weight = create_gaussian_weight_window(patch_size).to(self.device)

    @torch.no_grad()
    def predict_volume(
        self,
        volume_4ch: np.ndarray,
        crop_to_brain: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run patch-based sliding window inference on 4-channel MRI volume.

        Args:
            volume_4ch: Shape (4, D, H, W) float32 array
            crop_to_brain: If True, crops to non-zero brain bounding box first

        Returns:
            pred_mask: Shape (D, H, W) uint8 array with BraTS labels (0, 1, 2, 4)
            pred_probs: Shape (4, D, H, W) float32 probability map
        """
        orig_shape = volume_4ch.shape[1:]  # (D, H, W)
        D, H, W = orig_shape
        pd, ph, pw = self.patch_size

        if crop_to_brain:
            s_d, s_h, s_w = extract_brain_bbox(volume_4ch, margin=6)
            cropped_vol = volume_4ch[:, s_d, s_h, s_w]
        else:
            s_d, s_h, s_w = slice(0, D), slice(0, H), slice(0, W)
            cropped_vol = volume_4ch

        cD, cH, cW = cropped_vol.shape[1:]

        # Pad volume if smaller than patch size
        pad_d = max(0, pd - cD)
        pad_h = max(0, ph - cH)
        pad_w = max(0, pw - cW)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            cropped_vol = np.pad(
                cropped_vol,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
            cD, cH, cW = cropped_vol.shape[1:]

        # Stride computation
        step_d = max(1, int(pd * (1.0 - self.overlap)))
        step_h = max(1, int(ph * (1.0 - self.overlap)))
        step_w = max(1, int(pw * (1.0 - self.overlap)))

        # Patch coordinate ranges
        d_steps = list(range(0, cD - pd + 1, step_d))
        if d_steps[-1] != cD - pd:
            d_steps.append(cD - pd)

        h_steps = list(range(0, cH - ph + 1, step_h))
        if h_steps[-1] != cH - ph:
            h_steps.append(cH - ph)

        w_steps = list(range(0, cW - pw + 1, step_w))
        if w_steps[-1] != cW - pw:
            w_steps.append(cW - pw)

        # Output accumulators on CPU or GPU
        num_classes = 4
        prob_accum = torch.zeros((num_classes, cD, cH, cW), dtype=torch.float32, device=self.device)
        count_accum = torch.zeros((1, cD, cH, cW), dtype=torch.float32, device=self.device)

        weight = self.gaussian_weight.unsqueeze(0)  # Shape (1, pd, ph, pw)

        # Sliding window loop
        for d in d_steps:
            for h in h_steps:
                for w in w_steps:
                    patch = cropped_vol[:, d:d+pd, h:h+ph, w:w+pw]
                    patch_t = torch.from_numpy(patch).unsqueeze(0).to(self.device).float()

                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            logits = self.model(patch_t)
                    else:
                        logits = self.model(patch_t)

                    probs = F.softmax(logits, dim=1).squeeze(0)  # (C, pd, ph, pw)

                    # Weighted accumulation
                    prob_accum[:, d:d+pd, h:h+ph, w:w+pw] += probs * weight
                    count_accum[:, d:d+pd, h:h+ph, w:w+pw] += weight

        # Normalize probabilities
        prob_accum /= torch.clamp(count_accum, min=1e-5)

        # Remove padding if added
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            prob_accum = prob_accum[:, :cD - pad_d, :cH - pad_h, :cW - pad_w]

        # Put back into full volume shape
        full_probs = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=self.device)
        full_probs[0] = 1.0  # Background default
        full_probs[:, s_d, s_h, s_w] = prob_accum

        pred_classes = torch.argmax(full_probs, dim=0).cpu().numpy().astype(np.uint8)
        pred_probs_np = full_probs.cpu().numpy()

        # Map model classes (0, 1, 2, 3) back to BraTS raw labels (0, 1, 2, 4)
        raw_pred_mask = model_to_raw_labels(pred_classes)

        return raw_pred_mask, pred_probs_np


def compute_volumetrics(
    mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    affine: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive volumetric and geometric metrics from segmentation mask.

    BraTS labels:
      1: Necrotic & Non-Enhancing Core (NCR)
      2: Peritumoral Edema (ED)
      4: Enhancing Tumor (ET)
    """
    dx, dy, dz = voxel_spacing
    voxel_vol_cm3 = (dx * dy * dz) / 1000.0  # mm³ to cm³

    ncr_count = int(np.sum(mask == 1))
    ed_count = int(np.sum(mask == 2))
    et_count = int(np.sum(mask == 4))
    wt_count = ncr_count + ed_count + et_count
    tc_count = ncr_count + et_count

    vol_ncr = round(ncr_count * voxel_vol_cm3, 2)
    vol_ed = round(ed_count * voxel_vol_cm3, 2)
    vol_et = round(et_count * voxel_vol_cm3, 2)
    vol_wt = round(wt_count * voxel_vol_cm3, 2)
    vol_tc = round(tc_count * voxel_vol_cm3, 2)

    pct_ncr = round((ncr_count / wt_count * 100.0), 1) if wt_count > 0 else 0.0
    pct_ed = round((ed_count / wt_count * 100.0), 1) if wt_count > 0 else 0.0
    pct_et = round((et_count / wt_count * 100.0), 1) if wt_count > 0 else 0.0

    # Centroid computation in RAS physical space
    centroid_ras = {"x": 0.0, "y": 0.0, "z": 0.0}
    if wt_count > 0:
        tumor_coords = np.argwhere(mask > 0)
        # Note: if shape is (D, H, W) = (Z, Y, X), mean order is (cz, cy, cx)
        cz, cy, cx = np.mean(tumor_coords, axis=0)

        if affine is not None:
            # Transform voxel coordinate to RAS world coordinates
            voxel_homo = np.array([cx, cy, cz, 1.0])
            ras_point = affine @ voxel_homo
            centroid_ras = {
                "x": round(float(ras_point[0]), 1),
                "y": round(float(ras_point[1]), 1),
                "z": round(float(ras_point[2]), 1)
            }
        else:
            centroid_ras = {
                "x": round(float(cx * dx), 1),
                "y": round(float(cy * dy), 1),
                "z": round(float(cz * dz), 1)
            }

    hemisphere = "Midline / Central"
    if centroid_ras["x"] > 5.0:
        hemisphere = "Right Hemisphere"
    elif centroid_ras["x"] < -5.0:
        hemisphere = "Left Hemisphere"

    return {
        "total_tumor_cm3": vol_wt,
        "tumor_core_cm3": vol_tc,
        "enhancing_tumor_cm3": vol_et,
        "edema_cm3": vol_ed,
        "necrotic_core_cm3": vol_ncr,
        "breakdown": {
            "enhancing_tumor": {"volume_cm3": vol_et, "percentage": pct_et, "label": 4},
            "peritumoral_edema": {"volume_cm3": vol_ed, "percentage": pct_ed, "label": 2},
            "necrotic_core": {"volume_cm3": vol_ncr, "percentage": pct_ncr, "label": 1}
        },
        "voxel_counts": {
            "et": et_count,
            "ed": ed_count,
            "ncr": ncr_count,
            "wt": wt_count
        },
        "centroid_ras": centroid_ras,
        "affected_hemisphere": hemisphere,
        "has_tumor": wt_count > 0
    }


def save_segmentation_nifti(
    mask: np.ndarray,
    output_path: str,
    reference_nifti_path: Optional[str] = None,
    affine: Optional[np.ndarray] = None
) -> str:
    """
    Save predicted segmentation mask as a compressed NIfTI (.nii.gz) file.
    Transposes (D, H, W) back to standard BraTS NIfTI orientation (X, Y, Z) = (240, 240, 155).
    """
    if mask.ndim == 3 and mask.shape[0] == 155:
        # Transpose from (155, 240, 240) -> (240, 240, 155)
        nii_data = np.transpose(mask, (1, 2, 0)).astype(np.uint8)
    else:
        nii_data = mask.astype(np.uint8)

    if reference_nifti_path and os.path.exists(reference_nifti_path):
        ref_img = nib.load(reference_nifti_path)
        out_img = nib.Nifti1Image(nii_data, ref_img.affine, ref_img.header)
    else:
        if affine is None:
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
        out_img = nib.Nifti1Image(nii_data, affine)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nib.save(out_img, output_path)
    return output_path
