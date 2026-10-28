# EDI 1 - BrainVision 3D Medical Imaging Workstation & BraTS 2020 AI Segmentation

A browser-based neuroimaging workstation (Vanilla JS, Canvas, WebGL) and PyTorch deep learning pipeline for **Brain Tumor Segmentation using the BraTS 2020 dataset**.

Targeted & optimized for **NVIDIA GeForce RTX 3050 (6 GB VRAM)** using patch-based training, Automatic Mixed Precision (AMP), and small batch sizes.

---

## 🧠 BraTS 2020 Brain Tumor Segmentation

The workstation fully integrates the **BraTS 2020 Multi-Modal Benchmark** (Volumes: 240 &times; 240 &times; 155):

### 4 MRI Modalities
- **FLAIR** (Fluid-Attenuated Inversion Recovery): Visualizes peritumoral vasogenic edema (ED).
- **T1ce** (T1 with Gadolinium contrast): Visualizes hyperintense active enhancing tumor rim (ET).
- **T1** (T1-weighted native): Structural baseline, hypointense fluid/necrotic areas.
- **T2** (T2-weighted): Hyperintense fluid, ventricles, and edema boundaries.

### BraTS Segmentation Ground Truth Labels
- **Label 1 (NCR)**: Necrotic & Non-Enhancing Tumor Core &rarr; <span style="color:#06B6D4">**Cyan**</span>
- **Label 2 (ED)**: Peritumoral Vasogenic Edema &rarr; <span style="color:#F59E0B">**Amber / Yellow**</span>
- **Label 4 (ET)**: Active Enhancing Tumor &rarr; <span style="color:#EF4444">**Red**</span>
- **Label 0**: Background / Normal brain tissue

Composite clinical regions evaluated:
- **Whole Tumor (WT)**: $1 + 2 + 4$
- **Tumor Core (TC)**: $1 + 4$
- **Enhancing Tumor (ET)**: $4$

---

## ⚡ RTX 3050 (6 GB VRAM) Hardware Optimizations

| Strategy | Implementation | Why It Matters |
|---|---|---|
| **Patch-Based Sampling** | Default `(96, 96, 96)` or `(64, 64, 64)` | Fits 3D volumes into 6 GB VRAM without OOM |
| **Foreground-Biased Sampling** | 75% patches centered on tumor voxels | Solves severe class imbalance (<2% tumor volume) |
| **Mixed Precision (AMP)** | `torch.amp.autocast('cuda')` + `GradScaler` | 2x memory reduction and up to 3x speedup on Tensor Cores |
| **Small Batch Sizes** | Mini-batch $B=1$ or $2$ with gradient accumulation | Memory safety during backpropagation |
| **Instance Normalization** | `InstanceNorm3d(affine=True)` | Superior stability for small batch sizes where BatchNorm fails |
| **Sliding-Window Inference** | 50% overlap + 3D Gaussian blending | Reconstructs full 240&times;240&times;155 volumes with zero edge artifacts |

---

## 🚀 Quick Start Guide

### 1. Frontend Web Workstation (Live Server)
1. Open this folder in VS Code.
2. Start Live Server on `index.html` or run:
   ```bash
   npx live-server --port=8080
   # OR
   python -m http.server 8080
   ```
3. Open `http://127.0.0.1:8080/` in your browser.
4. Click **"Load BraTS 2020 Case"** in the bottom panel to immediately explore a 240&times;240&times;155 4-modality case with 3D Marching Cubes surface extraction!
5. Toggle between **FLAIR**, **T1ce**, **T1**, and **T2** in the *BraTS MRI Modality* selector.
6. Click **"Run AI Segmentation"** to trigger neural segmentation and live volumetric metrics in the *Analysis Suite* tab!

---

### 2. Backend Inference & Analysis API (FastAPI)

1. **Install requirements**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Launch the FastAPI Server**:
   ```bash
   python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
   ```

3. **Check API Telemetry**:
   ```bash
   curl http://127.0.0.1:8000/api/status
   ```
   Returns live GPU name, CUDA status, and VRAM telemetry (6.0 GB total).

4. **Available Endpoints**:
   - `GET /api/status`: System & GPU hardware health check
   - `POST /api/segment`: Upload multi-modal NIfTI files for 3D sliding-window segmentation
   - `POST /api/segment_synthetic`: Benchmark inference on a synthetic 240&times;240&times;155 BraTS volume
   - `GET /api/download_seg/{task_id}`: Download the generated `.nii.gz` segmentation mask

---

### 3. BraTS 2020 Model Training

To train the 3D Residual U-Net on your BraTS 2020 dataset:

```bash
python -m backend.train \
  --data_dir "path/to/MICCAI_BraTS2020_TrainingData" \
  --save_dir "checkpoints" \
  --patch_size 96 96 96 \
  --batch_size 1 \
  --grad_accum_steps 2 \
  --epochs 50 \
  --lr 2e-4 \
  --device cuda
```

To run a fast validation test with synthetic BraTS cases without downloading 20 GB:
```bash
python -m backend.train --generate_synthetic 2 --epochs 2 --patch_size 64 64 64 --batch_size 1 --device cuda
```

---

### 4. Running Unit Tests

Run the full automated test suite:
```bash
python -m pytest backend/test_pipeline.py -v
```

Tests verify:
- Synthetic BraTS 2020 volume generator & NIfTI serialization
- Raw to model label mapping ($0, 1, 2, 4 \leftrightarrow 0, 1, 2, 3$)
- 3D U-Net forward pass & parameter count
- Soft Dice + Cross-Entropy Loss computation
- BraTS validation metrics (WT, TC, ET Dice scores)
- Volumetric and centroid physical coordinate calculations

---

## 📁 Repository Structure

```
EDI-1/
├── index.html                   # Medical imaging workstation entry point
├── css/
│   └── style.css                # Medical workstation theme, HUD brackets & modality pills
├── js/
│   ├── lib/                     # Decompression & parsing libraries
│   │   ├── pako.min.js          # Gzip decompression
│   │   ├── nifti-reader.js      # NIfTI file format reader
│   │   ├── dicom-parser.min.js  # DICOM series parser
│   │   ├── three.min.js         # Three.js 3D WebGL library
│   │   └── OrbitControls.js     # 3D viewport navigation
│   ├── app.js                   # Application controller, BraTS generator & AI pipeline
│   ├── marching-cubes.js        # 3D Isosurface extraction algorithm
│   ├── three-viewer.js          # WebGL 3D brain & tumor isosurface renderer
│   ├── viewer.js                # Multi-planar 2D slice viewer & multiclass overlays
│   ├── nifti.js                 # Multi-modal NIfTI series extractor (BraTS support)
│   ├── dicom.js                 # DICOM parser
│   ├── state.js                 # Central state manager (modalities, masks, metrics)
│   └── upload.js                # Drag-and-drop & BraTS series ingestion
├── assets/
│   └── mni152.nii.gz            # Standard MNI152 template sample
├── backend/                     # PyTorch RTX 3050 BraTS 2020 Segmentation Pipeline
│   ├── requirements.txt         # Python dependencies
│   ├── dataset.py               # BraTS 2020 dataset loader, patch sampler & generator
│   ├── losses.py                # Soft Dice + CE Loss, WT/TC/ET Dice metrics
│   ├── inference.py             # Patch-based sliding window inferer & cm³ volumetrics
│   ├── train.py                 # Training script with AMP, small batch size & checkpointing
│   ├── server.py                # FastAPI inference server & GPU telemetry API
│   ├── test_pipeline.py         # Unit test suite
│   └── models/
│       ├── __init__.py
│       └── unet3d.py            # Memory-efficient 3D Residual U-Net with InstanceNorm3d
└── README.md
```

---

## 🔒 Security & Data Integrity

- Dataset files (`data/`, `datasets/`, `*.tar.gz`, `*.nii`, `*.nii.gz`), model checkpoints (`*.pth`, `*.pt`), and training weights are strictly excluded via `.gitignore` to prevent committing large data files.
- All existing features (MNI152 sample loading, single NIfTI/DICOM parsing, 3D WebGL controls, Window/Level adjustments) remain fully functional.
