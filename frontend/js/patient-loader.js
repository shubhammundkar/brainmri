/**
 * BrainVision 3D - Patient Multi-Modal Series Loader
 * Handles BraTS 2020 patient series: T1, T1ce, T2, FLAIR + ground truth _seg
 * Enforces strict validation:
 * 1. All four modalities (T1, T1ce, T2, FLAIR) must be present.
 * 2. All files must share the exact same patient ID (filename prefix).
 * 3. All volumes must share identical dimensions (nx, ny, nz) and voxel spacing (dx, dy, dz).
 * 4. _seg is strictly treated as ground-truth mask and NEVER loaded as an MRI volume.
 */

(function(root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.PatientLoader = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  'use strict';

  const MODALITIES = ['t1', 't1ce', 't2', 'flair'];

  /**
   * Detects the role (t1, t1ce, t2, flair, seg) and patient ID prefix from filename.
   * Matches formats like BraTS20_Training_369_t1.nii or patient123-t1ce.nii.gz.
   *
   * @param {string} fileName
   * @returns {{ role: string, patientId: string, originalName: string } | null}
   */
  function detectFileRoleAndPatient(fileName) {
    if (!fileName || typeof fileName !== 'string') return null;

    const lower = fileName.toLowerCase().trim();
    let base = lower;
    if (base.endsWith('.nii.gz')) {
      base = base.slice(0, -7);
    } else if (base.endsWith('.nii')) {
      base = base.slice(0, -4);
    } else {
      return null; // Not a NIfTI file
    }

    // Role patterns ordered with specific ones first (t1ce before t1)
    const patterns = [
      { role: 't1ce', regex: /^(.*?)[_.-]t1ce$/ },
      { role: 't1ce', regex: /^(.*?)[_.-]t1c$/ },
      { role: 'flair', regex: /^(.*?)[_.-]flair$/ },
      { role: 't2', regex: /^(.*?)[_.-]t2$/ },
      { role: 't1', regex: /^(.*?)[_.-]t1$/ },
      { role: 'seg', regex: /^(.*?)[_.-]seg$/ },
      { role: 'seg', regex: /^(.*?)[_.-]mask$/ }
    ];

    for (const p of patterns) {
      const match = base.match(p.regex);
      if (match) {
        // Recover original case for patient ID
        const prefixLen = match[1].length;
        const patientId = fileName.substring(0, prefixLen).replace(/[_.-]+$/, '') || match[1];
        return {
          role: p.role,
          patientId: patientId,
          originalName: fileName
        };
      }
    }

    return {
      role: 'unknown',
      patientId: base,
      originalName: fileName
    };
  }

  /**
   * Validates a collection of selected files before decompression.
   * Checks file presence, patient ID consistency, and ensures all 4 modalities exist.
   *
   * @param {File[]|{name: string}[]} files
   * @returns {{ valid: boolean, error?: string, patientId?: string, fileMap: Object, recognizedFiles: Object[] }}
   */
  function validatePatientFiles(files) {
    if (!files || files.length === 0) {
      return {
        valid: false,
        error: 'No files provided. Please select a patient folder or files.'
      };
    }

    const fileMap = {
      t1: null,
      t1ce: null,
      t2: null,
      flair: null,
      seg: null
    };

    const patientIds = new Set();
    const recognizedFiles = [];

    for (const file of files) {
      const detected = detectFileRoleAndPatient(file.name);
      if (!detected || detected.role === 'unknown') {
        continue; // Skip non-NIfTI files or unrelated files in directory
      }

      recognizedFiles.push({ file, ...detected });
      patientIds.add(detected.patientId);

      if (['t1', 't1ce', 't2', 'flair', 'seg'].includes(detected.role)) {
        fileMap[detected.role] = file;
      }
    }

    if (recognizedFiles.length === 0) {
      return {
        valid: false,
        error: 'No valid BraTS NIfTI files (.nii or .nii.gz) identified. Ensure filenames contain role suffixes (_t1, _t1ce, _t2, _flair).'
      };
    }

    // 1. Patient ID Consistency Check
    if (patientIds.size > 1) {
      const idList = Array.from(patientIds).join("', '");
      return {
        valid: false,
        error: `Patient ID Mismatch: Selected files belong to multiple patients ('${idList}'). All scans must belong to the same patient.`
      };
    }

    const patientId = Array.from(patientIds)[0];

    // 2. All 4 Modalities Required Check
    const missing = MODALITIES.filter(m => !fileMap[m]);
    if (missing.length > 0) {
      const missingLabels = missing.map(m => m.toUpperCase()).join(', ');
      return {
        valid: false,
        patientId,
        fileMap,
        missingModalities: missing,
        error: `Incomplete Patient Series for patient '${patientId}': Missing required modalities: [${missingLabels}]. BraTS analysis requires all 4 modalities (T1, T1ce, T2, FLAIR).`
      };
    }

    return {
      valid: true,
      patientId,
      fileMap,
      recognizedFiles
    };
  }

  /**
   * Helper to compute real tumor metrics from ground truth mask voxels
   */
  function computeGroundTruthVolumetrics(mask, nx, ny, nz, dx, dy, dz) {
    const voxelCm3 = (dx * dy * dz) / 1000.0;
    let ncrCount = 0, edCount = 0, etCount = 0;
    let sumX = 0, sumY = 0, sumZ = 0;

    for (let z = 0; z < nz; z++) {
      const zOff = z * nx * ny;
      for (let y = 0; y < ny; y++) {
        const rowOff = zOff + y * nx;
        for (let x = 0; x < nx; x++) {
          const l = mask[rowOff + x];
          if (l > 0) {
            sumX += x; sumY += y; sumZ += z;
            if (l === 1) ncrCount++;
            else if (l === 2) edCount++;
            else if (l === 4 || l === 3) etCount++;
          }
        }
      }
    }

    const wtCount = ncrCount + edCount + etCount;
    const volWt = parseFloat((wtCount * voxelCm3).toFixed(2));
    const volEt = parseFloat((etCount * voxelCm3).toFixed(2));
    const volEd = parseFloat((edCount * voxelCm3).toFixed(2));
    const volNcr = parseFloat((ncrCount * voxelCm3).toFixed(2));

    const pctEt = wtCount > 0 ? parseFloat((etCount / wtCount * 100).toFixed(1)) : 0;
    const pctEd = wtCount > 0 ? parseFloat((edCount / wtCount * 100).toFixed(1)) : 0;
    const pctNcr = wtCount > 0 ? parseFloat((ncrCount / wtCount * 100).toFixed(1)) : 0;

    const cx = wtCount > 0 ? (sumX / wtCount - nx / 2) * dx : 0;
    const cy = wtCount > 0 ? (sumY / wtCount - ny / 2) * dy : 0;
    const cz = wtCount > 0 ? (sumZ / wtCount - nz / 2) * dz : 0;

    return {
      has_tumor: wtCount > 0,
      total_tumor_cm3: volWt,
      enhancing_tumor_cm3: volEt,
      edema_cm3: volEd,
      necrotic_core_cm3: volNcr,
      affected_hemisphere: cx > 3 ? 'Right Hemisphere' : (cx < -3 ? 'Left Hemisphere' : 'Central / Bilateral'),
      centroid_ras: {
        x: parseFloat(cx.toFixed(1)),
        y: parseFloat(cy.toFixed(1)),
        z: parseFloat(cz.toFixed(1))
      },
      breakdown: {
        enhancing_tumor: { volume_cm3: volEt, percentage: pctEt, label: 4 },
        peritumoral_edema: { volume_cm3: volEd, percentage: pctEd, label: 2 },
        necrotic_core: { volume_cm3: volNcr, percentage: pctNcr, label: 1 }
      }
    };
  }

  /**
   * Executes the full Load Patient workflow:
   * 1. Validates files
   * 2. Decompresses & reads raw NIfTI data
   * 3. Validates dimensions & voxel spacing across all 4 scans
   * 4. Treats _seg strictly as ground truth (never loaded as an MRI scan volume)
   * 5. Commits clean patient dataset to AppState
   *
   * @param {File[]} files
   * @param {Function} [onProgress]
   */
  async function loadPatient(files, onProgress) {
    const validation = validatePatientFiles(files);
    if (!validation.valid) {
      if (typeof window !== 'undefined' && window.AppState) {
        window.AppState.setState(window.AppState.STATES.ERROR, {
          errorMessage: validation.error
        });
      }
      throw new Error(validation.error);
    }

    const { patientId, fileMap } = validation;

    if (typeof window !== 'undefined' && window.AppState) {
      window.AppState.setState(window.AppState.STATES.LOADING, {
        fileName: `${patientId} (BraTS Multi-Modal)`,
        fileSize: (files.reduce((a, f) => a + (f.size || 0), 0) / (1024 * 1024)).toFixed(2) + ' MB'
      });
    }

    if (onProgress) onProgress('Reading and decompressing modalities...');

    // Read the 4 mandatory MRI modalities
    const rawResults = {};
    for (const mod of MODALITIES) {
      if (onProgress) onProgress(`Decompressing ${mod.toUpperCase()} modality...`);
      rawResults[mod] = await window.NiftiParser.readNiftiRaw(fileMap[mod]);
    }

    // Dimension & Voxel Spacing Consistency Verification
    const refMod = 'flair';
    const refMeta = rawResults[refMod].meta;
    const refNx = refMeta.nx, refNy = refMeta.ny, refNz = refMeta.nz;
    const refDx = refMeta.dx, refDy = refMeta.dy, refDz = refMeta.dz;

    for (const mod of MODALITIES) {
      const meta = rawResults[mod].meta;
      if (meta.nx !== refNx || meta.ny !== refNy || meta.nz !== refNz) {
        const dimErr = `Dimension Mismatch: Modality '${mod.toUpperCase()}' has shape (${meta.nx} × ${meta.ny} × ${meta.nz}), but reference '${refMod.toUpperCase()}' has (${refNx} × ${refNy} × ${refNz}). All modalities must share identical dimensions.`;
        if (typeof window !== 'undefined' && window.AppState) {
          window.AppState.setState(window.AppState.STATES.ERROR, { errorMessage: dimErr });
        }
        throw new Error(dimErr);
      }

      // Voxel spacing tolerance: 0.05 mm
      const tol = 0.05;
      if (Math.abs(meta.dx - refDx) > tol || Math.abs(meta.dy - refDy) > tol || Math.abs(meta.dz - refDz) > tol) {
        const spaceErr = `Voxel Spacing Mismatch: Modality '${mod.toUpperCase()}' has spacing (${meta.dx.toFixed(2)}, ${meta.dy.toFixed(2)}, ${meta.dz.toFixed(2)}) mm, but reference '${refMod.toUpperCase()}' has (${refDx.toFixed(2)}, ${refDy.toFixed(2)}, ${refDz.toFixed(2)}) mm.`;
        if (typeof window !== 'undefined' && window.AppState) {
          window.AppState.setState(window.AppState.STATES.ERROR, { errorMessage: spaceErr });
        }
        throw new Error(spaceErr);
      }
    }

    // Ground Truth Segmentation Mask Handling
    let segMask = null;
    let segMetrics = null;

    if (fileMap.seg) {
      if (onProgress) onProgress('Reading Ground Truth segmentation mask (_seg)...');
      const segRes = await window.NiftiParser.readNiftiRaw(fileMap.seg);

      if (segRes.meta.nx !== refNx || segRes.meta.ny !== refNy || segRes.meta.nz !== refNz) {
        const segErr = `Ground Truth Mask Mismatch: Segmentation mask shape (${segRes.meta.nx} × ${segRes.meta.ny} × ${segRes.meta.nz}) does not match MRI scan shape (${refNx} × ${refNy} × ${refNz}).`;
        if (typeof window !== 'undefined' && window.AppState) {
          window.AppState.setState(window.AppState.STATES.ERROR, { errorMessage: segErr });
        }
        throw new Error(segErr);
      }

      segMask = new Uint8Array(segRes.voxels.length);
      for (let i = 0; i < segRes.voxels.length; i++) {
        segMask[i] = Math.round(segRes.voxels[i]);
      }
      segMetrics = computeGroundTruthVolumetrics(segMask, refNx, refNy, refNz, refDx, refDy, refDz);
    }

    // Commit to AppState
    if (typeof window !== 'undefined' && window.AppState) {
      // 1. Metadata from primary reference
      const primaryMeta = Object.assign({}, refMeta, {
        fileName: `${patientId}_series`,
        patientId: patientId
      });
      window.AppState.setDatasetMetadata(primaryMeta);

      // 2. Keep original File objects for backend API upload
      window.AppState.setPatientFiles(patientId, fileMap);

      // 3. Store the 4 MRI scan volumes (NEVER include _seg as an MRI volume)
      window.AppState.setModalities({
        t1: rawResults.t1.voxels,
        t1ce: rawResults.t1ce.voxels,
        t2: rawResults.t2.voxels,
        flair: rawResults.flair.voxels
      }, 'flair');

      // 4. If _seg is present, register it as separate ground truth mask
      if (segMask) {
        window.AppState.setSegmentationMask(segMask, {
          1: 'Necrotic Core (NCR)',
          2: 'Peritumoral Edema (ED)',
          4: 'Enhancing Tumor (ET)'
        }, segMetrics, 'ground_truth');
      } else {
        window.AppState.clearSegmentation();
      }

      window.AppState.setState(window.AppState.STATES.LOADED);
    }

    return {
      patientId,
      modalities: MODALITIES,
      hasSeg: !!segMask,
      metrics: segMetrics
    };
  }

  return {
    MODALITIES,
    detectFileRoleAndPatient,
    validatePatientFiles,
    computeGroundTruthVolumetrics,
    loadPatient
  };
});
