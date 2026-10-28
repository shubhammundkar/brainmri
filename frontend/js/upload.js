/**
 * BrainVision 3D - File Upload, Drag-and-Drop & Sample Dataset Loader
 */

window.UploadManager = (function() {
  'use strict';

  let selectedFiles = [];

  function initUpload(elements) {
    const { dropzone, fileInput, folderInput, browseBtn, loadPatientBtn, sampleBtn } = elements;

    if (loadPatientBtn && folderInput) {
      loadPatientBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        folderInput.click();
      });
    }

    if (browseBtn && fileInput) {
      browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }

    if (folderInput) {
      folderInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
          handleFilesChosen(files);
        }
      });
    }

    if (sampleBtn) {
      sampleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        loadSampleNifti();
      });
    }

    if (dropzone) {
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
    const isNiftiCollection = files.some(f => {
      const lower = f.name.toLowerCase();
      return lower.endsWith('.nii') || lower.endsWith('.nii.gz');
    });

    try {
      // 1. If multiple files with NIfTI are dropped/selected, run the PatientLoader flow
      if (isNiftiCollection && (files.length > 1 || files.some(f => window.PatientLoader && window.PatientLoader.detectFileRoleAndPatient(f.name)?.role !== 'unknown'))) {
        if (window.PatientLoader) {
          await window.PatientLoader.loadPatient(files);
          return;
        }
      }

      // 2. Single NIfTI scan fallback
      const fname = first.name.toLowerCase();
      if (fname.endsWith('.nii') || fname.endsWith('.nii.gz')) {
        window.AppState.setState(window.AppState.STATES.LOADING, {
          fileName: first.name,
          fileSize: (first.size / (1024 * 1024)).toFixed(2) + ' MB'
        });
        await window.NiftiParser.parseNiftiFile(first);
        window.AppState.setState(window.AppState.STATES.LOADED);
        return;
      }

      // 3. DICOM series
      if (fname.endsWith('.dcm') || isMultiDicom) {
        window.AppState.setState(window.AppState.STATES.LOADING, {
          fileName: isMultiDicom ? `${files.length} DICOM Series (${first.name})` : first.name,
          fileSize: (files.reduce((a, f) => a + f.size, 0) / (1024 * 1024)).toFixed(2) + ' MB'
        });
        await window.DicomParserModule.parseDicomFiles(files);
        window.AppState.setState(window.AppState.STATES.LOADED);
        return;
      }

      throw new Error('Unsupported format. Please select BraTS NIfTI files (.nii or .nii.gz) or DICOM files (.dcm).');
    } catch (err) {
      console.error('File Parse Error:', err);
      window.AppState.setState(window.AppState.STATES.ERROR, {
        errorMessage: err.message || 'Failed to read dataset files.'
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
