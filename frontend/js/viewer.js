/**
 * BrainVision 3D - Multi-Planar 2D Slice Viewer & Real-Time Mask Overlay Renderer
 * High-performance clinical voxel extraction and authentic anatomical visualization
 */

window.MriViewer = (function() {
  'use strict';

  // Offscreen helper canvases for slice rasterization
  const offscreen = {
    axial: document.createElement('canvas'),
    sagittal: document.createElement('canvas'),
    coronal: document.createElement('canvas')
  };

  const offscreenOverlay = {
    axial: document.createElement('canvas'),
    sagittal: document.createElement('canvas'),
    coronal: document.createElement('canvas')
  };

  // Crosshair visibility state
  const hudCrosshairs = {
    axial: true,
    sagittal: true,
    coronal: true
  };

  // High-fidelity synthetic brain cache
  let syntheticVolume = null;

  // Active tumor component visibility filters
  const componentVisibility = {
    1: true, // NCR
    2: true, // ED
    3: true, // ET (model)
    4: true  // ET (BraTS raw)
  };

  function setComponentVisibility(labelId, visible) {
    componentVisibility[labelId] = !!visible;
    if (labelId === 3 || labelId === 4) {
      componentVisibility[3] = !!visible;
      componentVisibility[4] = !!visible;
    }
  }

  function isComponentVisible(label) {
    return componentVisibility[label] !== false;
  }

  function initSyntheticBrain(nx = 215, ny = 256, nz = 215) {
    if (syntheticVolume) return syntheticVolume;
    const voxels = new Float32Array(nx * ny * nz);
    const cx = nx / 2, cy = ny / 2, cz = nz / 2;

    for (let z = 0; z < nz; z++) {
      const zOffset = z * nx * ny;
      const nzNorm = (z - cz) / (nz * 0.44);

      for (let y = 0; y < ny; y++) {
        const rowOffset = zOffset + y * nx;
        const nyNorm = (y - cy) / (ny * 0.46);

        for (let x = 0; x < nx; x++) {
          const nxNorm = (x - cx) / (nx * 0.38);
          const rSq = nxNorm * nxNorm + nyNorm * nyNorm + nzNorm * nzNorm;
          let val = 0;

          // Scalp, Skull Bone & Extracranial Tissue
          if (rSq >= 0.88 && rSq <= 0.98) {
            val = 220 + 35 * Math.sin(x * 0.4 + y * 0.4);
          } else if (rSq > 0.82 && rSq < 0.88) {
            val = 20; // CSF
          } else if (rSq <= 0.82) {
            const isCerebellum = (nyNorm < -0.15 && nzNorm < -0.15 && rSq < 0.75);
            const isBrainstem = (Math.abs(nxNorm) < 0.18 && nyNorm > -0.15 && nyNorm < 0.25 && nzNorm < -0.10);

            const sulciNoise = Math.sin(x * 0.18 + Math.cos(y * 0.15) * 1.5) *
                               Math.cos(y * 0.18 + Math.sin(z * 0.15) * 1.5) *
                               Math.sin(z * 0.16 + Math.cos(x * 0.15));
            
            const fissure = Math.exp(-Math.pow(nxNorm * 28.0, 2));

            const vX = Math.abs(nxNorm) * 3.2;
            const vY = (nyNorm + 0.05) * 2.2;
            const vZ = (nzNorm - 0.05) * 2.6;
            const isVentricle = (vX > 0.10 && vX < 0.60) && (vY * vY + vZ * vZ < 0.42);

            const isCorpusCallosum = (Math.abs(nxNorm) < 0.35) && 
                                     (nyNorm * nyNorm + Math.pow(nzNorm - 0.18, 2) < 0.35) &&
                                     (nyNorm * nyNorm + Math.pow(nzNorm - 0.18, 2) > 0.20);

            if (isVentricle) {
              val = 15;
            } else if (isCorpusCallosum) {
              val = 225 + sulciNoise * 15;
            } else if (isCerebellum) {
              const folia = Math.sin(z * 0.8) * 40;
              val = 150 + folia;
            } else if (isBrainstem) {
              val = 180 + Math.sin(y * 0.3) * 20;
            } else {
              const isDeepWhite = rSq < 0.45;
              const baseIntensity = isDeepWhite ? 200 : 135;
              const convoluted = baseIntensity + sulciNoise * 45;
              val = Math.max(25, Math.min(255, convoluted * (1.0 - fissure * 0.85)));
            }
          }

          voxels[rowOffset + x] = val;
        }
      }
    }
    syntheticVolume = { nx, ny, nz, voxels };
    return syntheticVolume;
  }

  function renderSlice(canvas, planeType) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    const state = window.AppState.state;
    let meta = state.dataset;
    let voxels = state.volume;

    // If no custom dataset loaded, use synthetic volume
    if (!voxels || meta.nz === 0) {
      const syn = initSyntheticBrain();
      meta = { nx: syn.nx, ny: syn.ny, nz: syn.nz, maxVoxel: 255, minVoxel: 0 };
      voxels = syn.voxels;
    }

    ctx.clearRect(0, 0, w, h);

    if (!state.viewport.brainVisible) {
      return;
    }

    const nx = meta.nx;
    const ny = meta.ny;
    const nz = meta.nz;
    const ww = state.viewport.windowWidth || 1500;
    const wl = state.viewport.windowLevel || 750;
    const minVis = wl - ww / 2;

    let sliceW = 0;
    let sliceH = 0;

    if (planeType === 'axial') {
      sliceW = nx; sliceH = ny;
    } else if (planeType === 'sagittal') {
      sliceW = ny; sliceH = nz;
    } else if (planeType === 'coronal') {
      sliceW = nx; sliceH = nz;
    }

    if (sliceW === 0 || sliceH === 0) return;

    // Prepare offscreen canvas
    const offCanvas = offscreen[planeType];
    if (offCanvas.width !== sliceW || offCanvas.height !== sliceH) {
      offCanvas.width = sliceW;
      offCanvas.height = sliceH;
    }

    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(sliceW, sliceH);
    const data = imgData.data;

    const sliceIdx = state.viewport.slice[planeType] !== undefined 
      ? state.viewport.slice[planeType] 
      : Math.floor(sliceH / 2);

    const sliceVoxelCount = nx * ny;
    let ptr = 0;

    if (planeType === 'axial') {
      const z = Math.min(Math.max(0, sliceIdx), nz - 1);
      const zOffset = z * sliceVoxelCount;

      for (let y = 0; y < ny; y++) {
        const rowOffset = zOffset + y * nx;
        for (let x = 0; x < nx; x++) {
          const val = voxels[rowOffset + x];
          let gray = 0;
          if (val > minVis) {
            gray = Math.min(255, Math.max(0, Math.round(((val - minVis) / ww) * 255)));
          }
          data[ptr] = gray;
          data[ptr + 1] = gray;
          data[ptr + 2] = gray;
          data[ptr + 3] = 255;
          ptr += 4;
        }
      }

    } else if (planeType === 'sagittal') {
      const x = Math.min(Math.max(0, sliceIdx), nx - 1);

      for (let pz = 0; pz < nz; pz++) {
        const z = nz - 1 - pz;
        const zOffset = z * sliceVoxelCount;

        for (let y = 0; y < ny; y++) {
          const val = voxels[zOffset + y * nx + x];
          let gray = 0;
          if (val > minVis) {
            gray = Math.min(255, Math.max(0, Math.round(((val - minVis) / ww) * 255)));
          }
          data[ptr] = gray;
          data[ptr + 1] = gray;
          data[ptr + 2] = gray;
          data[ptr + 3] = 255;
          ptr += 4;
        }
      }

    } else if (planeType === 'coronal') {
      const y = Math.min(Math.max(0, sliceIdx), ny - 1);

      for (let pz = 0; pz < nz; pz++) {
        const z = nz - 1 - pz;
        const zOffset = z * sliceVoxelCount + y * nx;

        for (let x = 0; x < nx; x++) {
          const val = voxels[zOffset + x];
          let gray = 0;
          if (val > minVis) {
            gray = Math.min(255, Math.max(0, Math.round(((val - minVis) / ww) * 255)));
          }
          data[ptr] = gray;
          data[ptr + 1] = gray;
          data[ptr + 2] = gray;
          data[ptr + 3] = 255;
          ptr += 4;
        }
      }
    }

    offCtx.putImageData(imgData, 0, 0);

    // Draw transformed slice on target canvas
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const zoom = state.viewport.zoom[planeType] || 1.0;
    const pan = state.viewport.pan[planeType] || { x: 0, y: 0 };

    ctx.translate(w / 2 + pan.x, h / 2 + pan.y);
    ctx.scale(zoom, zoom);

    const scale = Math.min((w * 0.88) / sliceW, (h * 0.88) / sliceH);
    const drawW = sliceW * scale;
    const drawH = sliceH * scale;

    ctx.drawImage(offCanvas, -drawW / 2, -drawH / 2, drawW, drawH);
    ctx.restore();

    // Render Crosshairs if active
    if (hudCrosshairs[planeType]) {
      drawHudCrosshairs(ctx, w, h);
    }
  }

  /**
   * Render real AI segmentation mask slice on 2D overlay canvas
   */
  function renderRealSegmentationOverlay(overlayCanvas, planeType) {
    if (!overlayCanvas) return;
    const ctx = overlayCanvas.getContext('2d');
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const state = window.AppState.state;
    if (!state.viewport.tumorVisible) return;

    const mask = state.segmentationMask;
    const meta = state.dataset;
    if (!mask || meta.nx === 0) return;

    const nx = meta.nx;
    const ny = meta.ny;
    const nz = meta.nz;

    let sliceW = 0;
    let sliceH = 0;
    if (planeType === 'axial') { sliceW = nx; sliceH = ny; }
    else if (planeType === 'sagittal') { sliceW = ny; sliceH = nz; }
    else if (planeType === 'coronal') { sliceW = nx; sliceH = nz; }

    if (sliceW === 0 || sliceH === 0) return;

    const offOverlay = offscreenOverlay[planeType];
    if (offOverlay.width !== sliceW || offOverlay.height !== sliceH) {
      offOverlay.width = sliceW;
      offOverlay.height = sliceH;
    }

    const offCtx = offOverlay.getContext('2d');
    const imgData = offCtx.createImageData(sliceW, sliceH);
    const data = imgData.data;

    const sliceIdx = state.viewport.slice[planeType] !== undefined 
      ? state.viewport.slice[planeType] 
      : Math.floor(sliceH / 2);

    const sliceVoxelCount = nx * ny;
    let ptr = 0;
    let hasTumorOnSlice = false;

    if (planeType === 'axial') {
      const z = Math.min(Math.max(0, sliceIdx), nz - 1);
      const zOffset = z * sliceVoxelCount;

      for (let y = 0; y < ny; y++) {
        const rowOffset = zOffset + y * nx;
        for (let x = 0; x < nx; x++) {
          const label = mask[rowOffset + x];
          if (label > 0 && isComponentVisible(label)) {
            hasTumorOnSlice = true;
            if (label === 1) { // NCR (Cyan)
              data[ptr] = 6; data[ptr+1] = 182; data[ptr+2] = 212; data[ptr+3] = 190;
            } else if (label === 2) { // ED (Amber)
              data[ptr] = 245; data[ptr+1] = 158; data[ptr+2] = 11; data[ptr+3] = 160;
            } else { // ET (Red)
              data[ptr] = 239; data[ptr+1] = 68; data[ptr+2] = 68; data[ptr+3] = 220;
            }
          }
          ptr += 4;
        }
      }
    } else if (planeType === 'sagittal') {
      const x = Math.min(Math.max(0, sliceIdx), nx - 1);

      for (let pz = 0; pz < nz; pz++) {
        const z = nz - 1 - pz;
        const zOffset = z * sliceVoxelCount;

        for (let y = 0; y < ny; y++) {
          const label = mask[zOffset + y * nx + x];
          if (label > 0 && isComponentVisible(label)) {
            hasTumorOnSlice = true;
            if (label === 1) {
              data[ptr] = 6; data[ptr+1] = 182; data[ptr+2] = 212; data[ptr+3] = 190;
            } else if (label === 2) {
              data[ptr] = 245; data[ptr+1] = 158; data[ptr+2] = 11; data[ptr+3] = 160;
            } else {
              data[ptr] = 239; data[ptr+1] = 68; data[ptr+2] = 68; data[ptr+3] = 220;
            }
          }
          ptr += 4;
        }
      }
    } else if (planeType === 'coronal') {
      const y = Math.min(Math.max(0, sliceIdx), ny - 1);

      for (let pz = 0; pz < nz; pz++) {
        const z = nz - 1 - pz;
        const zOffset = z * sliceVoxelCount + y * nx;

        for (let x = 0; x < nx; x++) {
          const label = mask[zOffset + x];
          if (label > 0 && isComponentVisible(label)) {
            hasTumorOnSlice = true;
            if (label === 1) {
              data[ptr] = 6; data[ptr+1] = 182; data[ptr+2] = 212; data[ptr+3] = 190;
            } else if (label === 2) {
              data[ptr] = 245; data[ptr+1] = 158; data[ptr+2] = 11; data[ptr+3] = 160;
            } else {
              data[ptr] = 239; data[ptr+1] = 68; data[ptr+2] = 68; data[ptr+3] = 220;
            }
          }
          ptr += 4;
        }
      }
    }

    if (!hasTumorOnSlice) return;

    offCtx.putImageData(imgData, 0, 0);

    ctx.save();
    const zoom = state.viewport.zoom[planeType] || 1.0;
    const pan = state.viewport.pan[planeType] || { x: 0, y: 0 };

    ctx.translate(w / 2 + pan.x, h / 2 + pan.y);
    ctx.scale(zoom, zoom);

    const scale = Math.min((w * 0.88) / sliceW, (h * 0.88) / sliceH);
    const drawW = sliceW * scale;
    const drawH = sliceH * scale;

    ctx.drawImage(offOverlay, -drawW / 2, -drawH / 2, drawW, drawH);
    ctx.restore();
  }

  function drawHudCrosshairs(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    ctx.beginPath();
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Center target reticle
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 3.5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  function toggleCrosshair(planeType) {
    if (hudCrosshairs[planeType] !== undefined) {
      hudCrosshairs[planeType] = !hudCrosshairs[planeType];
    }
  }

  /**
   * Compute Voxel coordinates and intensity under mouse cursor for HUD telemetry
   */
  function inspectVoxelAtCanvas(canvas, planeType, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return null;

    const state = window.AppState.state;
    const meta = state.dataset;
    const voxels = state.volume;
    const mask = state.segmentationMask;

    if (!voxels || meta.nx === 0) return null;

    const nx = meta.nx;
    const ny = meta.ny;
    const nz = meta.nz;
    const dx = meta.dx || 1.0;
    const dy = meta.dy || 1.0;
    const dz = meta.dz || 1.0;

    let sliceW = 0, sliceH = 0;
    if (planeType === 'axial') { sliceW = nx; sliceH = ny; }
    else if (planeType === 'sagittal') { sliceW = ny; sliceH = nz; }
    else if (planeType === 'coronal') { sliceW = nx; sliceH = nz; }

    const zoom = state.viewport.zoom[planeType] || 1.0;
    const pan = state.viewport.pan[planeType] || { x: 0, y: 0 };
    const scale = Math.min((w * 0.88) / sliceW, (h * 0.88) / sliceH);

    // Map screen mouse pos to slice image coordinates
    const centerX = w / 2 + pan.x;
    const centerY = h / 2 + pan.y;

    const drawW = sliceW * scale * zoom;
    const drawH = sliceH * scale * zoom;

    const imgX = (x - (centerX - drawW / 2)) / (drawW / sliceW);
    const imgY = (y - (centerY - drawH / 2)) / (drawH / sliceH);

    if (imgX < 0 || imgX >= sliceW || imgY < 0 || imgY >= sliceH) return null;

    const vx = Math.floor(imgX);
    const vy = Math.floor(imgY);
    const sliceIdx = state.viewport.slice[planeType] || 0;

    let targetX = 0, targetY = 0, targetZ = 0;
    if (planeType === 'axial') {
      targetX = vx;
      targetY = vy;
      targetZ = sliceIdx;
    } else if (planeType === 'sagittal') {
      targetX = sliceIdx;
      targetY = vx;
      targetZ = nz - 1 - vy;
    } else if (planeType === 'coronal') {
      targetX = vx;
      targetY = sliceIdx;
      targetZ = nz - 1 - vy;
    }

    if (targetX < 0 || targetX >= nx || targetY < 0 || targetY >= ny || targetZ < 0 || targetZ >= nz) return null;

    const index = targetZ * nx * ny + targetY * nx + targetX;
    const intensity = Math.round(voxels[index]);
    const label = mask ? mask[index] : 0;

    let regionName = 'Brain Tissue';
    if (intensity < (meta.minVoxel + 10)) regionName = 'Background / Air';
    else if (label === 1) regionName = 'Necrotic Core (NCR)';
    else if (label === 2) regionName = 'Peritumoral Edema (ED)';
    else if (label === 3 || label === 4) regionName = 'Enhancing Tumor (ET)';

    return {
      voxel: { x: targetX, y: targetY, z: targetZ },
      physical: {
        x: (targetX * dx).toFixed(1),
        y: (targetY * dy).toFixed(1),
        z: (targetZ * dz).toFixed(1)
      },
      intensity: intensity,
      region: regionName,
      label: label
    };
  }

  return {
    renderSlice,
    renderRealSegmentationOverlay,
    setComponentVisibility,
    toggleCrosshair,
    inspectVoxelAtCanvas
  };
})();
