/**
 * BrainVision 3D - Central Application State Manager
 * Holds UI states, MRI dataset metadata, 3D voxel volume buffer,
 * and 3D AI segmentation masks with multiclass label structures.
 */

window.AppState = (function() {
  'use strict';

  // Application States
  const STATES = {
    NO_MRI: 'NO_MRI',
    SELECTED: 'SELECTED',
    LOADING: 'LOADING',
    LOADED: 'LOADED',
    ERROR: 'ERROR'
  };

  const state = {
    currentState: STATES.NO_MRI,
    errorMessage: '',
    dataset: {
      fileName: null,
      fileSize: null,
      fileType: null,
      nx: 0,
      ny: 0,
      nz: 0,
      dx: 1.0,
      dy: 1.0,
      dz: 1.0,
      datatype: 'Unknown',
      minVoxel: 0,
      maxVoxel: 255
    },
    // Raw 3D volume voxel array: Float32Array of size (nx * ny * nz)
    volume: null,

    // BraTS 2020 Multi-Modal volumes: { t1: Float32Array, t1ce: ..., t2: ..., flair: ... }
    modalities: {
      t1: null,
      t1ce: null,
      t2: null,
      flair: null
    },
    currentModality: 't1',

    // Real 3D AI Segmentation Mask: Uint8Array of size (nx * ny * nz)
    // BraTS 2020 Labels: 0: background, 1: NCR (Necrotic Core), 2: ED (Edema), 4: ET (Enhancing Tumor)
    segmentationMask: null,
    segmentationMetrics: null,
    segmentationLabels: {
      1: 'Necrotic Core (NCR)',
      2: 'Peritumoral Edema (ED)',
      4: 'Enhancing Tumor (ET)'
    },
    
    // Viewport control states
    viewport: {
      slice: { axial: 0, sagittal: 0, coronal: 0 },
      maxSlice: { axial: 0, sagittal: 0, coronal: 0 },
      zoom: { axial: 1.0, sagittal: 1.0, coronal: 1.0, '3d': 1.0 },
      pan: {
        axial: { x: 0, y: 0 },
        sagittal: { x: 0, y: 0 },
        coronal: { x: 0, y: 0 }
      },
      windowWidth: 400,
      windowLevel: 40,
      mriOpacity: 1.0,
      brainVisible: true,
      tumorVisible: false,
      brain3DVisible: true,
      tumor3DVisible: true,
      brainOpacity: 0.35,
      tumorOpacity: 0.95,
      maximizedPanel: null
    }
  };

  const listeners = [];

  function subscribe(fn) {
    listeners.push(fn);
  }

  function notify() {
    listeners.forEach(fn => fn(state));
  }

  function setState(newState, payload = {}) {
    state.currentState = newState;
    if (payload.errorMessage) state.errorMessage = payload.errorMessage;
    notify();
  }

  function setDatasetMetadata(meta) {
    Object.assign(state.dataset, meta);
    
    // Set default slice indices to middle slice
    state.viewport.maxSlice.axial = Math.max(0, meta.nz - 1);
    state.viewport.maxSlice.sagittal = Math.max(0, meta.nx - 1);
    state.viewport.maxSlice.coronal = Math.max(0, meta.ny - 1);

    state.viewport.slice.axial = Math.floor(meta.nz / 2);
    state.viewport.slice.sagittal = Math.floor(meta.nx / 2);
    state.viewport.slice.coronal = Math.floor(meta.ny / 2);

    // Auto-compute Window Width (WW) and Window Level (WL) from voxel intensity distribution
    const range = Math.max(1, meta.maxVoxel - meta.minVoxel);
    state.viewport.windowWidth = Math.round(range * 0.8);
    state.viewport.windowLevel = Math.round(meta.minVoxel + range * 0.4);

    notify();
  }

  function setVolumeData(voxels) {
    state.volume = voxels;
    state.segmentationMask = null; // Clear previous segmentation on new dataset
    notify();
  }

  function setModalities(modalitiesMap, active = 't1') {
    state.modalities = Object.assign({}, state.modalities, modalitiesMap);
    const chosen = state.modalities[active] ? active : Object.keys(modalitiesMap)[0];
    if (chosen && state.modalities[chosen]) {
      state.currentModality = chosen;
      state.volume = state.modalities[chosen];
    }
    notify();
  }

  function setModality(modName) {
    const key = modName.toLowerCase();
    if (state.modalities[key]) {
      state.currentModality = key;
      state.volume = state.modalities[key];
      notify();
      return true;
    }
    return false;
  }

  function setSegmentationMask(mask, labels = null, metrics = null) {
    state.segmentationMask = mask;
    if (labels) state.segmentationLabels = labels;
    if (metrics) state.segmentationMetrics = metrics;
    state.viewport.tumorVisible = true;
    state.viewport.tumor3DVisible = true;
    notify();
  }

  function clearSegmentation() {
    state.segmentationMask = null;
    state.viewport.tumorVisible = false;
    notify();
  }

  function resetViewport() {
    if (state.dataset.nz > 0) {
      state.viewport.slice.axial = Math.floor(state.dataset.nz / 2);
      state.viewport.slice.sagittal = Math.floor(state.dataset.nx / 2);
      state.viewport.slice.coronal = Math.floor(state.dataset.ny / 2);
    }
    state.viewport.zoom = { axial: 1.0, sagittal: 1.0, coronal: 1.0, '3d': 1.0 };
    state.viewport.pan = {
      axial: { x: 0, y: 0 },
      sagittal: { x: 0, y: 0 },
      coronal: { x: 0, y: 0 }
    };
    notify();
  }

  return {
    STATES,
    state,
    subscribe,
    setState,
    setDatasetMetadata,
    setVolumeData,
    setModalities,
    setModality,
    setSegmentationMask,
    clearSegmentation,
    resetViewport
  };
})();
