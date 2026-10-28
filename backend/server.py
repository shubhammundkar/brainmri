"""
BrainVision 3D - FastAPI Inference & Analysis Server for BraTS 2020 Tumor Segmentation
Hardware Optimized: NVIDIA RTX 3050 (6 GB VRAM)

Endpoints:
- GET  /api/status: System & GPU telemetry (VRAM, CUDA, active model)
- POST /api/segment: Multi-modal NIfTI segmentation with sliding-window inference
- POST /api/segment_synthetic: Immediate benchmark inference on synthetic BraTS anatomy
- GET  /api/download_seg/{task_id}: Download generated NIfTI segmentation mask (.nii.gz)
"""

import os
import io
import uuid
import tempfile
from typing import Optional, Dict, Tuple

import numpy as np
import nibabel as nib
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .models.unet3d import ResidualUNet3D
from .dataset import normalize_intensity, generate_synthetic_case
from .inference import SlidingWindowInferer, compute_volumetrics, save_segmentation_nifti

app = FastAPI(
    title="BrainVision 3D - BraTS Tumor Segmentation API",
    version="2.0.0",
    description="Clinical multi-modal brain tumor segmentation workstation backend"
)

# Enable CORS for browser client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & inferer cache
MODEL = None
INFERER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR = tempfile.mkdtemp(prefix="brainvision_seg_")


def get_gpu_telemetry() -> Dict:
    """Retrieve GPU memory and hardware telemetry."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free_gb = total_gb - reserved_gb
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total_gb, 2),
            "vram_allocated_gb": round(allocated_gb, 2),
            "vram_free_gb": round(free_gb, 2)
        }
    return {
        "cuda_available": False,
        "device_name": "CPU Host",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_free_gb": 0.0
    }


def load_model(checkpoint_path: Optional[str] = None):
    """Initialize or load 3D U-Net model and sliding window inferer."""
    global MODEL, INFERER, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    base_filters = 16  # Optimal for 6 GB VRAM
    model = ResidualUNet3D(in_channels=4, out_channels=4, base_filters=base_filters)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        # Default checkpoints lookup
        default_ckpts = [
            "checkpoints/best_model.pth",
            "checkpoints/latest_model.pth",
            os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
        ]
        loaded = False
        for path in default_ckpts:
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=DEVICE)
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                model.load_state_dict(state_dict)
                print(f"Loaded default model weights from: {path}")
                loaded = True
                break
        if not loaded:
            print("Notice: Initialized ResidualUNet3D with initial weights (trained weights not found yet).")

    model.to(DEVICE).eval()
    MODEL = model
    INFERER = SlidingWindowInferer(
        model=model,
        patch_size=(96, 96, 96),
        overlap=0.5,
        device=DEVICE,
        use_amp=(DEVICE == "cuda")
    )
    return INFERER


@app.on_event("startup")
async def startup_event():
    load_model()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mount static frontend assets
for folder in ["css", "js", "assets"]:
    f_path = os.path.join(ROOT_DIR, folder)
    if os.path.exists(f_path):
        app.mount(f"/{folder}", StaticFiles(directory=f_path), name=folder)


@app.get("/", response_class=FileResponse)
async def serve_workstation():
    """Serve the BrainVision 3D browser workstation directly from the server."""
    index_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return FileResponse(os.path.join(ROOT_DIR, "frontend", "index.html"))


@app.get("/server-status", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Interactive server status, GPU hardware telemetry & BraTS AI control dashboard."""
    t = get_gpu_telemetry()
    vram_pct = round((t["vram_allocated_gb"] / max(0.1, t["vram_total_gb"])) * 100, 1) if t["cuda_available"] else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BrainVision 3D - Server Telemetry & BraTS Control Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #090C10;
      color: #E2E8F0;
      font-family: 'Inter', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    .header {{
      background: #111622;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding: 18px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-title {{
      font-size: 18px;
      font-weight: 700;
      color: #F8FAFC;
      letter-spacing: -0.02em;
    }}
    .brand-sub {{
      font-size: 12px;
      color: #00F0FF;
      font-family: 'JetBrains Mono', monospace;
    }}
    .header-links {{
      display: flex;
      gap: 10px;
    }}
    .btn {{
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
      border: 1px solid transparent;
    }}
    .btn-cyan {{
      background: #00F0FF;
      color: #090C10;
    }}
    .btn-cyan:hover {{
      background: #38BDF8;
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
    }}
    .btn-outline {{
      background: rgba(255, 255, 255, 0.04);
      color: #E2E8F0;
      border-color: rgba(255, 255, 255, 0.12);
    }}
    .btn-outline:hover {{
      background: rgba(255, 255, 255, 0.08);
      color: #FFF;
    }}
    .main-container {{
      max-width: 1200px;
      margin: 32px auto;
      padding: 0 24px;
      flex: 1;
      width: 100%;
    }}
    .grid-2col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #111622;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}
    .card-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94A3B8;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
    }}
    .pill-green {{
      background: rgba(16, 185, 129, 0.15);
      color: #10B981;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 11px;
    }}
    .pill-cyan {{
      background: rgba(0, 240, 255, 0.15);
      color: #00F0FF;
      border: 1px solid rgba(0, 240, 255, 0.3);
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 11px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 13.5px;
    }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-key {{ color: #94A3B8; }}
    .metric-val {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #F1F5F9; }}
    .progress-bar-bg {{
      background: rgba(255, 255, 255, 0.08);
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 12px;
    }}
    .progress-bar-fill {{
      background: linear-gradient(90deg, #00F0FF, #10B981);
      height: 100%;
      transition: width 0.3s ease;
    }}
    .hero-benchmark {{
      background: linear-gradient(135deg, rgba(0, 240, 255, 0.05), rgba(41, 121, 255, 0.05));
      border: 1px solid rgba(0, 240, 255, 0.2);
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .benchmark-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .code-box {{
      background: #090C10;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      color: #A5F3FC;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
    }}
    .footer {{
      padding: 20px;
      text-align: center;
      font-size: 12px;
      color: #64748B;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }}
  </style>
</head>
<body>
  <header class="header">
    <div class="brand">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00F0FF" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
      <div>
        <div class="brand-title">BrainVision 3D Server</div>
        <div class="brand-sub">FastAPI BraTS 2020 AI Segmentation Engine</div>
      </div>
    </div>
    <div class="header-links">
      <a href="/" class="btn btn-cyan">Open 3D Workstation &rarr;</a>
      <a href="/docs" class="btn btn-outline" target="_blank">Swagger API Docs</a>
      <a href="/api/status" class="btn btn-outline" target="_blank">JSON Status</a>
    </div>
  </header>

  <main class="main-container">
    <div class="grid-2col">
      <!-- GPU Hardware Card -->
      <div class="card">
        <div class="card-title">
          <span>GPU Hardware Telemetry</span>
          <span class="pill-green">{"CUDA Active" if t["cuda_available"] else "CPU Mode"}</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">GPU Device</span>
          <span class="metric-val">{t["device_name"]}</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Total VRAM</span>
          <span class="metric-val">{t["vram_total_gb"]} GB</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">VRAM Allocated</span>
          <span class="metric-val">{t["vram_allocated_gb"]} GB ({vram_pct}%)</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">VRAM Free</span>
          <span class="metric-val">{t["vram_free_gb"]} GB</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Precision Mode</span>
          <span class="metric-val">Automatic Mixed Precision (AMP fp16)</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {max(5.0, vram_pct)}%"></div>
        </div>
      </div>

      <!-- AI Model Specs Card -->
      <div class="card">
        <div class="card-title">
          <span>BraTS 2020 Model Specs</span>
          <span class="pill-cyan">Residual 3D U-Net</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Architecture</span>
          <span class="metric-val">3D Residual U-Net + InstanceNorm3d</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Input Modalities (4ch)</span>
          <span class="metric-val">FLAIR, T1ce, T1, T2</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Target Labels (BraTS)</span>
          <span class="metric-val">1: NCR | 2: ED | 4: ET</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Volume Dimensions</span>
          <span class="metric-val">240 &times; 240 &times; 155 voxels</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Sliding Window Patch</span>
          <span class="metric-val">(96, 96, 96) &bull; 50% Overlap Gaussian</span>
        </div>
      </div>
    </div>

    <!-- Live Benchmark Tool -->
    <div class="hero-benchmark">
      <div class="benchmark-header">
        <div>
          <h3 style="color:#FFF; font-size:16px;">Live 3D Sliding-Window Benchmark</h3>
          <p style="color:#94A3B8; font-size:13px; margin-top:4px;">Execute sliding-window inference on synthetic 240&times;240&times;155 volume on the RTX 3050.</p>
        </div>
        <button class="btn btn-cyan" id="btn-run-benchmark" onclick="runBenchmark()">Run AI Benchmark</button>
      </div>
      <div class="code-box" id="benchmark-output">Click "Run AI Benchmark" to test sliding window inference across all 4 BraTS modalities...</div>
    </div>
  </main>

  <footer class="footer">
    BrainVision 3D Workstation &bull; NVIDIA RTX 3050 (6 GB VRAM) &bull; BraTS 2020 Multimodal AI Segmentation
  </footer>

  <script>
    async function runBenchmark() {{
      const btn = document.getElementById('btn-run-benchmark');
      const box = document.getElementById('benchmark-output');
      btn.disabled = true;
      btn.textContent = 'Running Inference on RTX 3050...';
      box.textContent = 'Executing patch-based sliding window inference (50% overlap, 3D Gaussian weighting, AMP)...\\n';

      const t0 = performance.now();
      try {{
        const res = await fetch('/api/segment_synthetic', {{ method: 'POST' }});
        const data = await res.json();
        const duration = ((performance.now() - t0) / 1000).toFixed(2);
        box.textContent = `Inference Completed in ${{duration}}s!\\n\\n` + JSON.stringify(data, null, 2);
      }} catch (err) {{
        box.textContent = 'Error running benchmark: ' + err.message;
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'Run AI Benchmark';
      }}
    }}
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/api/status")
async def get_status():
    """System health check and GPU hardware status."""
    telemetry = get_gpu_telemetry()
    return {
        "status": "online",
        "service": "BrainVision 3D BraTS Segmentation Engine",
        "version": "2.0.0",
        "telemetry": telemetry,
        "model": {
            "name": "Residual 3D U-Net (BraTS 2020)",
            "base_filters": 16,
            "patch_size": [96, 96, 96],
            "input_modalities": ["T1", "T1ce", "T2", "FLAIR"],
            "labels": {
                "1": "Necrotic & Non-Enhancing Core (NCR)",
                "2": "Peritumoral Edema (ED)",
                "4": "Enhancing Tumor (ET)"
            }
        }
    }


def _read_nifti_bytes(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Parse NIfTI bytes into normalized numpy array, affine, and zooms."""
    import gzip
    if filename.endswith(".gz"):
        try:
            decompressed = gzip.decompress(file_bytes)
            fh = nib.FileHolder(fileobj=io.BytesIO(decompressed))
            nii = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        except Exception:
            fh = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
            nii = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    else:
        fh = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
        nii = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

    data = nii.get_fdata(dtype=np.float32)
    # Convert BraTS (240, 240, 155) -> (155, 240, 240) = (D, H, W)
    if data.ndim == 3 and data.shape == (240, 240, 155):
        data = np.transpose(data, (2, 0, 1))

    zooms = nii.header.get_zooms()[:3]
    return normalize_intensity(data), nii.affine, zooms


@app.post("/api/segment")
async def run_segmentation(
    t1: Optional[UploadFile] = File(None),
    t1ce: Optional[UploadFile] = File(None),
    t2: Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
    single_file: Optional[UploadFile] = File(None)
):
    """
    Execute 3D patch-based segmentation on uploaded MRI modalities.
    Accepts all 4 BraTS modalities or a single volume with pseudo-multimodal expansion.
    """
    global INFERER
    if INFERER is None:
        INFERER = load_model()

    try:
        modality_arrays = []
        affine = None
        zooms = (1.0, 1.0, 1.0)

        # Case 1: Separate 4 modalities provided
        if t1 and t1ce and t2 and flair:
            for upload_f in [t1, t1ce, t2, flair]:
                b = await upload_f.read()
                arr, aff, z = _read_nifti_bytes(b, upload_f.filename)
                modality_arrays.append(arr)
                if affine is None:
                    affine = aff
                    zooms = z
            volume_4ch = np.stack(modality_arrays, axis=0)

        # Case 2: Single multi-channel or single modality provided
        elif single_file is not None or t1 is not None or flair is not None:
            f = single_file or t1 or flair
            b = await f.read()
            arr, aff, z = _read_nifti_bytes(b, f.filename)
            affine = aff
            zooms = z

            if arr.ndim == 4 and arr.shape[0] == 4:
                volume_4ch = arr
            else:
                # Replicate single scan into 4 channels
                volume_4ch = np.stack([arr, arr, arr, arr], axis=0)
        else:
            raise HTTPException(status_code=400, detail="Please upload either all 4 BraTS modalities (t1, t1ce, t2, flair) or a single_file.")

        # Run sliding-window inference strictly within RTX 3050 6GB VRAM
        pred_mask, _ = INFERER.predict_volume(volume_4ch, crop_to_brain=True)

        # Volumetrics calculation
        volumetrics = compute_volumetrics(pred_mask, voxel_spacing=zooms, affine=affine)

        # Save output NIfTI to temporary storage for download
        task_id = str(uuid.uuid4())
        out_nii_path = os.path.join(TEMP_DIR, f"{task_id}_seg.nii.gz")
        save_segmentation_nifti(pred_mask, out_nii_path, affine=affine)

        D, H, W = pred_mask.shape

        return {
            "status": "success",
            "task_id": task_id,
            "dimensions": {"depth": D, "height": H, "width": W},
            "voxel_spacing": {"dx": zooms[0], "dy": zooms[1], "dz": zooms[2]},
            "volumetrics": volumetrics,
            "download_url": f"/api/download_seg/{task_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/api/segment_synthetic")
async def run_synthetic_benchmark():
    """
    Execute instant sliding-window benchmark on realistic synthetic BraTS volume.
    Useful for demonstration and automated testing.
    """
    global INFERER
    if INFERER is None:
        INFERER = load_model()

    temp_case_dir = os.path.join(TEMP_DIR, "synthetic_case")
    case_path = generate_synthetic_case(temp_case_dir, subject_id="BraTS20_Demo")

    # Load the 4 modalities
    mods = []
    affine = None
    for m in ['t1', 't1ce', 't2', 'flair']:
        p = os.path.join(case_path, f"BraTS20_Demo_{m}.nii.gz")
        nii = nib.load(p)
        arr = nii.get_fdata(dtype=np.float32)
        if arr.shape == (240, 240, 155):
            arr = np.transpose(arr, (2, 0, 1))
        if affine is None:
            affine = nii.affine
        mods.append(normalize_intensity(arr))

    volume_4ch = np.stack(mods, axis=0)

    # Perform sliding window inference
    pred_mask, _ = INFERER.predict_volume(volume_4ch, crop_to_brain=True)
    volumetrics = compute_volumetrics(pred_mask, voxel_spacing=(1.0, 1.0, 1.0), affine=affine)

    task_id = str(uuid.uuid4())
    out_path = os.path.join(TEMP_DIR, f"{task_id}_seg.nii.gz")
    save_segmentation_nifti(pred_mask, out_path, affine=affine)

    return {
        "status": "success",
        "benchmark": "Synthetic BraTS 2020 Case (240x240x155)",
        "task_id": task_id,
        "volumetrics": volumetrics,
        "download_url": f"/api/download_seg/{task_id}"
    }


@app.get("/api/download_seg/{task_id}")
async def download_segmentation(task_id: str):
    """Download segmentation mask file (.nii.gz)."""
    file_path = os.path.join(TEMP_DIR, f"{task_id}_seg.nii.gz")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Segmentation file not found or expired.")
    return FileResponse(
        file_path,
        media_type="application/gzip",
        filename=f"brats_segmentation_{task_id[:8]}.nii.gz"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
