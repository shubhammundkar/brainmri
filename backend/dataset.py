"""
BrainVision 3D - BraTS 2020 Medical Dataset Loader & Preprocessing Pipeline
Target Hardware: NVIDIA RTX 3050 (6 GB VRAM)

Features:
- Handles 4 MRI modalities: T1, T1ce, T2, FLAIR + Seg
- BraTS raw label mapping: 0=Background, 1=NCR, 2=ED, 4=ET <-> Model: 0, 1, 2, 3
- Non-zero brain z-score intensity normalization
- Patch-based sampling with foreground-biased centering
- On-the-fly 3D medical augmentations
- Synthetic BraTS volume generator for unit testing & demonstration
"""

import os
import glob
import math
import random
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

# BraTS 2020 Raw Dimensions
BRATS_SHAPE = (240, 240, 155)  # (X, Y, Z) in NIfTI or (Z, Y, X) when transposed

# Modality filenames convention in BraTS 2020
MODALITIES = ['t1', 't1ce', 't2', 'flair']


def raw_to_model_labels(seg: np.ndarray) -> np.ndarray:
    """
    Map BraTS raw segmentation labels:
      0: Background -> 0
      1: Necrotic & Non-Enhancing Core (NCR) -> 1
      2: Peritumoral Edema (ED) -> 2
      4: Enhancing Tumor (ET) -> 3
    """
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def model_to_raw_labels(seg: np.ndarray) -> np.ndarray:
    """
    Map model predicted classes back to official BraTS raw labels:
      0 -> 0 (Background)
      1 -> 1 (NCR)
      2 -> 2 (ED)
      3 -> 4 (ET)
    """
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 3] = 4
    return out


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """
    Z-score intensity normalization on non-zero brain voxels.
    Background (intensity == 0) remains 0.0.
    """
    out = np.zeros_like(volume, dtype=np.float32)
    mask = volume > 0
    if np.any(mask):
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        if std > 1e-6:
            out[mask] = (volume[mask] - mean) / std
        else:
            out[mask] = volume[mask] - mean
    return out


def extract_brain_bbox(volume_4ch: np.ndarray, margin: int = 5) -> Tuple[slice, slice, slice]:
    """
    Compute minimal 3D bounding box enclosing the brain to remove empty background padding.
    volume_4ch: shape (4, D, H, W)
    """
    # Any channel non-zero
    mask = np.any(volume_4ch > 0, axis=0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return slice(None), slice(None), slice(None)

    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0) + 1

    D, H, W = mask.shape
    d_min = max(0, d_min - margin)
    h_min = max(0, h_min - margin)
    w_min = max(0, w_min - margin)

    d_max = min(D, d_max + margin)
    h_max = min(H, h_max + margin)
    w_max = min(W, w_max + margin)

    return slice(d_min, d_max), slice(h_min, h_max), slice(w_min, w_max)


def sample_patch_coords(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    seg: Optional[np.ndarray] = None,
    p_foreground: float = 0.75
) -> Tuple[int, int, int]:
    """
    Sample top-left corner coordinates (d, h, w) for a 3D patch.
    If p_foreground and seg contains tumor voxels, center the patch around a tumor voxel.
    """
    D, H, W = volume_shape
    pd, ph, pw = patch_size

    # Maximum starting index
    max_d = max(0, D - pd)
    max_h = max(0, H - ph)
    max_w = max(0, W - pw)

    if seg is not None and random.random() < p_foreground:
        fg_indices = np.argwhere(seg > 0)
        if len(fg_indices) > 0:
            center_idx = random.randint(0, len(fg_indices) - 1)
            cd, ch, cw = fg_indices[center_idx]
            # Center the patch on this foreground voxel with boundary clamping
            d = min(max(0, cd - pd // 2), max_d)
            h = min(max(0, ch - ph // 2), max_h)
            w = min(max(0, cw - pw // 2), max_w)
            return int(d), int(h), int(w)

    d = random.randint(0, max_d) if max_d > 0 else 0
    h = random.randint(0, max_h) if max_h > 0 else 0
    w = random.randint(0, max_w) if max_w > 0 else 0
    return int(d), int(h), int(w)


def apply_augmentations_3d(
    image: np.ndarray,
    seg: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply random 3D spatial & intensity augmentations to image (4, D, H, W) and seg (D, H, W).
    """
    # 1. Random 3D Flips
    for axis_idx, axis in enumerate([1, 2, 3]):  # D, H, W axes of image
        if random.random() < 0.5:
            image = np.flip(image, axis=axis)
            if seg is not None:
                seg = np.flip(seg, axis=axis_idx)

    # 2. Random 90-degree rotations in H-W (axial) plane
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        image = np.rot90(image, k, axes=(2, 3))
        if seg is not None:
            seg = np.rot90(seg, k, axes=(1, 2))

    # 3. Random intensity scale and shift
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.1, 0.1)
        image = image * scale + shift

    # 4. Small Gaussian noise
    if random.random() < 0.2:
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = image + noise

    image = np.ascontiguousarray(image)
    if seg is not None:
        seg = np.ascontiguousarray(seg)

    return image, seg


class BraTS2020Dataset(Dataset):
    """
    PyTorch Dataset for BraTS 2020 Multi-Modal Tumor Segmentation.

    Args:
        data_dir: Root directory containing subject folders (e.g. MICCAI_BraTS2020_TrainingData)
        patch_size: 3D patch dimensions (D, H, W), default: (96, 96, 96) for RTX 3050 6GB
        is_training: If True, applies random foreground-biased patch sampling and augmentations
        samples_per_volume: Number of patches extracted per volume per epoch
        p_foreground: Probability of sampling a patch centered on tumor tissue (default: 0.75)
    """
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        is_training: bool = True,
        samples_per_volume: int = 4,
        p_foreground: float = 0.75
    ):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.is_training = is_training
        self.samples_per_volume = samples_per_volume
        self.p_foreground = p_foreground

        self.subject_paths = self._find_subjects()
        if len(self.subject_paths) == 0:
            print(f"Warning: No BraTS subject folders found in '{data_dir}'. Use generate_synthetic_case() for testing.")

    def _find_subjects(self) -> List[str]:
        """Find valid BraTS subject directories."""
        if not os.path.exists(self.data_dir):
            return []
        subjects = []
        for entry in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, entry)
            if os.path.isdir(full_path) and not entry.startswith('.'):
                # Verify at least one modality exists
                has_mod = any(len(glob.glob(os.path.join(full_path, f"*{m}.nii*"))) > 0 for m in MODALITIES)
                if has_mod:
                    subjects.append(full_path)
        return sorted(subjects)

    def __len__(self) -> int:
        return len(self.subject_paths) * (self.samples_per_volume if self.is_training else 1)

    def _load_subject_volume(self, subject_dir: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """Load 4 modalities and segmentation for a given subject."""
        modality_arrays = []
        affine = None
        header_meta = {}

        for mod in MODALITIES:
            pattern = os.path.join(subject_dir, f"*{mod}.nii*")
            files = glob.glob(pattern)
            if len(files) == 0:
                raise FileNotFoundError(f"Missing {mod} modality in {subject_dir}")
            nii = nib.load(files[0])
            data = nii.get_fdata(dtype=np.float32)
            if affine is None:
                affine = nii.affine
                header_meta = {'zooms': nii.header.get_zooms()}
            # BraTS NIfTI is typically (240, 240, 155). Transpose to (155, 240, 240) -> (D, H, W)
            if data.ndim == 3 and data.shape == (240, 240, 155):
                data = np.transpose(data, (2, 0, 1))
            data = normalize_intensity(data)
            modality_arrays.append(data)

        volume_4ch = np.stack(modality_arrays, axis=0)  # Shape: (4, D, H, W)

        # Load segmentation mask
        seg_pattern = os.path.join(subject_dir, "*seg.nii*")
        seg_files = glob.glob(seg_pattern)
        seg_data = None
        if len(seg_files) > 0:
            seg_nii = nib.load(seg_files[0])
            raw_seg = seg_nii.get_fdata(dtype=np.float32)
            if raw_seg.ndim == 3 and raw_seg.shape == (240, 240, 155):
                raw_seg = np.transpose(raw_seg, (2, 0, 1))
            seg_data = raw_to_model_labels(raw_seg)

        return volume_4ch, seg_data, {'affine': affine, 'meta': header_meta}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj_idx = idx // self.samples_per_volume if self.is_training else idx
        subj_dir = self.subject_paths[subj_idx]

        volume_4ch, seg_data, info = self._load_subject_volume(subj_dir)
        D, H, W = volume_4ch.shape[1:]

        if self.is_training:
            # Sample patch
            d, h, w = sample_patch_coords(
                (D, H, W), self.patch_size, seg=seg_data, p_foreground=self.p_foreground
            )
            pd, ph, pw = self.patch_size
            patch_img = volume_4ch[:, d:d+pd, h:h+ph, w:w+pw]
            patch_seg = seg_data[d:d+pd, h:h+ph, w:w+pw] if seg_data is not None else None

            # Apply augmentations
            patch_img, patch_seg = apply_augmentations_3d(patch_img, patch_seg)

            item = {
                'image': torch.from_numpy(patch_img).float(),
                'subject': os.path.basename(subj_dir)
            }
            if patch_seg is not None:
                item['seg'] = torch.from_numpy(patch_seg).long()
            return item
        else:
            # Return full volume for evaluation / inference
            item = {
                'image': torch.from_numpy(volume_4ch).float(),
                'subject': os.path.basename(subj_dir)
            }
            if seg_data is not None:
                item['seg'] = torch.from_numpy(seg_data).long()
            return item


def generate_synthetic_case(
    output_dir: str,
    subject_id: str = "BraTS20_Synthetic_001",
    shape: Tuple[int, int, int] = (240, 240, 155)
) -> str:
    """
    Generate realistic synthetic BraTS 2020 multi-modal volume and ground-truth segmentation.
    Creates:
      <subject_id>_t1.nii.gz
      <subject_id>_t1ce.nii.gz
      <subject_id>_t2.nii.gz
      <subject_id>_flair.nii.gz
      <subject_id>_seg.nii.gz
    """
    os.makedirs(output_dir, exist_ok=True)
    subj_folder = os.path.join(output_dir, subject_id)
    os.makedirs(subj_folder, exist_ok=True)

    nx, ny, nz = shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Coordinate grids
    x = np.linspace(-1, 1, nx, dtype=np.float32)[:, None, None]
    y = np.linspace(-1, 1, ny, dtype=np.float32)[None, :, None]
    z = np.linspace(-1, 1, nz, dtype=np.float32)[None, None, :]

    r_brain = (x / 0.8)**2 + (y / 0.88)**2 + (z / 0.75)**2
    brain_mask = r_brain <= 1.0

    # Base brain parenchymal tissue intensities
    parenchyma = np.where(brain_mask, 100.0 + 30.0 * np.sin(x * 5) * np.cos(y * 5), 0.0).astype(np.float32)

    # Ventricles (CSF)
    r_vent = (np.abs(x) / 0.15)**2 + ((y + 0.05) / 0.35)**2 + (z / 0.3)**2
    vent_mask = brain_mask & (r_vent <= 1.0) & (np.abs(x) > 0.02)

    # Lesion location (right frontal-parietal region)
    lx, ly, lz = 0.28, 0.15, 0.20
    lr_x, lr_y, lr_z = 0.18, 0.15, 0.16
    r_tumor = ((x - lx) / lr_x)**2 + ((y - ly) / lr_y)**2 + ((z - lz) / lr_z)**2

    # BraTS Labels:
    # 0: Background
    # 1: Necrotic Core (NCR) inside r_tumor < 0.32
    # 4: Enhancing Tumor (ET) ring: 0.32 <= r_tumor < 0.65
    # 2: Peritumoral Edema (ED) outer rim: 0.65 <= r_tumor <= 1.0
    seg = np.zeros(shape, dtype=np.uint8)
    seg[brain_mask & (r_tumor <= 1.0)] = 2        # Edema (ED = 2)
    seg[brain_mask & (r_tumor < 0.65)] = 4        # Enhancing Tumor (ET = 4)
    seg[brain_mask & (r_tumor < 0.30)] = 1        # Necrotic Core (NCR = 1)

    # Modality specific signals:
    # T1: CSF dark, brain gray/white, tumor isointense/hypointense
    t1 = parenchyma.copy()
    t1[vent_mask] = 20.0
    t1[seg == 1] = 30.0   # NCR dark
    t1[seg == 2] = 70.0   # Edema slightly dark on T1
    t1[seg == 4] = 95.0

    # T1ce: Enhancing tumor (ET = 4) shows marked hyperintensity
    t1ce = t1.copy()
    t1ce[seg == 4] = 220.0  # Hyperintense contrast uptake

    # T2: CSF hyperintense, Edema hyperintense
    t2 = parenchyma * 0.8
    t2[vent_mask] = 240.0
    t2[seg == 2] = 210.0   # Edema bright on T2
    t2[seg == 1] = 160.0
    t2[seg == 4] = 140.0

    # FLAIR: CSF suppressed (dark), Edema and tumor core bright
    flair = parenchyma * 0.85
    flair[vent_mask] = 15.0  # CSF suppressed
    flair[seg == 2] = 230.0  # Edema hyperintense on FLAIR
    flair[seg == 4] = 180.0
    flair[seg == 1] = 120.0

    # Standard 1x1x1 mm affine matrix
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    # Save NIfTI files
    files = {
        f"{subject_id}_t1.nii.gz": t1,
        f"{subject_id}_t1ce.nii.gz": t1ce,
        f"{subject_id}_t2.nii.gz": t2,
        f"{subject_id}_flair.nii.gz": flair,
        f"{subject_id}_seg.nii.gz": seg
    }

    for fname, arr in files.items():
        fpath = os.path.join(subj_folder, fname)
        img = nib.Nifti1Image(arr, affine)
        nib.save(img, fpath)

    return subj_folder
