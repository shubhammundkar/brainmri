/**
 * BrainVision 3D - File Upload, Drag-and-Drop & Sample Dataset Loader
 */

window.UploadManager = (function() {
  'use strict';

  let selectedFiles = [];

  function initUpload(elements) {
    const { dropzone, fileInput, browseBtn, sampleBtn } = elements;

    if (browseBtn && fileInput) {
      browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }

    if (sampleBtn) {
      sampleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        loadSampleNifti();
      });
    }

    if (dropzone && fileInput) {
      ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault(); e.stopPropagation();
          dropzone.classList.add('dragover');
        }, false);
      });

      ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault(); e.stopPropagation();
          dropzone.classList.remove('dragover');
        }, false);
      });

      dropzone.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
          handleFilesChosen(files);
        }
      });
    }

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
          handleFilesChosen(files);
        }
      });
    }
  }

  async function handleFilesChosen(files) {
    selectedFiles = files;
    const first = files[0];
    const isMultiDicom = files.length > 1 && files.every(f => f.name.toLowerCase().endsWith('.dcm'));
    const isMultiNifti = files.length > 1 && files.some(f => f.name.toLowerCase().includes('.nii'));

    const nameDisplay = isMultiDicom 
      ? `${files.length} DICOM Series (${first.name} ...)`
      : (isMultiNifti ? `BraTS Multi-Modal Series (${files.length} files)` : first.name);
    const sizeDisplay = (files.reduce((a, f) => a + f.size, 0) / (1024 * 1024)).toFixed(2) + ' MB';

    window.AppState.setState(window.AppState.STATES.LOADING, {
      fileName: nameDisplay,
      fileSize: sizeDisplay
    });

    try {
      const fname = first.name.toLowerCase();
      if (isMultiNifti) {
        await window.NiftiParser.parseBraTSSeries(files);
      } else if (fname.endsWith('.nii') || fname.endsWith('.nii.gz')) {
        await window.NiftiParser.parseNiftiFile(first);
      } else if (fname.endsWith('.dcm') || isMultiDicom) {
        await window.DicomParserModule.parseDicomFiles(files);
      } else {
        throw new Error('Unsupported format. Please provide .nii, .nii.gz or .dcm files.');
      }

      window.AppState.setState(window.AppState.STATES.LOADED);
    } catch (err) {
      console.error('File Parse Error:', err);
      window.AppState.setState(window.AppState.STATES.ERROR, {
        errorMessage: err.message || 'Failed to read dataset file.'
      });
    }
  }

  async function loadSampleNifti() {
    window.AppState.setState(window.AppState.STATES.LOADING, {
      fileName: 'mni152.nii.gz',
      fileSize: '4.13 MB'
    });

    try {
      let response = await fetch('assets/mni152.nii.gz').catch(() => null);
      if (!response || !response.ok) {
        response = await fetch('frontend/assets/mni152.nii.gz').catch(() => null);
      }
      if (!response || !response.ok) {
        response = await fetch('./assets/mni152.nii.gz').catch(() => null);
      }
      if (!response || !response.ok) {
        throw new Error('Sample MNI152 file not found on server.');
      }
      const blob = await response.blob();
      const file = new File([blob], 'mni152.nii.gz', { type: 'application/gzip' });
      await window.NiftiParser.parseNiftiFile(file);
      window.AppState.setState(window.AppState.STATES.LOADED);
    } catch (err) {
      console.warn('Sample load fallback:', err);
      // If fetch fails in file:// protocol, update state to demo mode
      window.AppState.setState(window.AppState.STATES.LOADED);
    }
  }

  return {
    initUpload,
    loadSampleNifti,
    getSelectedFiles: () => selectedFiles
  };
})();
