/**
 * BrainVision 3D - NIfTI Medical Image Parser Module (.nii & .nii.gz)
 * Supports standard anatomical scans and BraTS 2020 Multi-Modal series (T1, T1ce, T2, FLAIR, Seg)
 */

window.NiftiParser = (function() {
  'use strict';

  /**
   * Low-level helper to decompress & read NIfTI header and voxels from a File object
   */
  async function readNiftiRaw(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async function(e) {
        try {
          let buffer = e.target.result;
          buffer = await window.nifti.decompressIfNeeded(buffer, file.name);

          if (!window.nifti.isNIFTI(buffer)) {
            return reject(new Error(`Invalid NIfTI signature in '${file.name}'.`));
          }

          const header = window.nifti.readHeader(buffer);
          const voxels = window.nifti.readImage(header, buffer);

          let minV = Infinity, maxV = -Infinity;
          for (let i = 0; i < voxels.length; i++) {
            const v = voxels[i];
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
          }
          if (minV === Infinity) minV = 0;
          if (maxV === -Infinity) maxV = 255;

          const meta = {
            fileName: file.name,
            fileSize: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
            fileType: file.name.endsWith('.gz') ? 'Compressed NIfTI (.nii.gz)' : 'NIfTI (.nii)',
            nx: header.nx,
            ny: header.ny,
            nz: header.nz,
            dx: parseFloat((header.pixDims[0] || 1.0).toFixed(2)),
            dy: parseFloat((header.pixDims[1] || 1.0).toFixed(2)),
            dz: parseFloat((header.pixDims[2] || 1.0).toFixed(2)),
            datatype: `Float32 (${header.bitsPerVoxel || 32}-bit)`,
            minVoxel: Math.round(minV),
            maxVoxel: Math.round(maxV)
          };

          resolve({ header, voxels, meta });
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = () => reject(new Error(`Failed to read file '${file.name}'.`));
      reader.readAsArrayBuffer(file);
    });
  }

  /**
   * Parse a single NIfTI file (standard anatomical scan or standalone segmentation)
   */
  async function parseNiftiFile(file) {
    const { voxels, meta } = await readNiftiRaw(file);

    // If file is standalone BraTS segmentation mask (_seg.nii.gz)
    const lower = file.name.toLowerCase();
    if (lower.includes('seg') || lower.includes('mask')) {
      // Check if current dataset matches dimensions
      const cur = window.AppState.state.dataset;
      if (cur.nx === meta.nx && cur.ny === meta.ny && cur.nz === meta.nz) {
        window.AppState.setSegmentationMask(voxels, {
          1: 'Necrotic Core (NCR)',
          2: 'Peritumoral Edema (ED)',
          4: 'Enhancing Tumor (ET)'
        });
        return meta;
      }
    }

    window.AppState.setDatasetMetadata(meta);
    window.AppState.setVolumeData(voxels);
    return meta;
  }

  /**
   * Parse BraTS 2020 Multi-Modal series (T1, T1ce, T2, FLAIR, plus optional _seg)
   */
  async function parseBraTSSeries(files) {
    if (!files || files.length === 0) return null;

    // Categorize files by BraTS modality
    let t1File = null;
    let t1ceFile = null;
    let t2File = null;
    let flairFile = null;
    let segFile = null;
    const others = [];

    files.forEach(f => {
      const name = f.name.toLowerCase();
      if (name.includes('t1ce') || name.includes('t1c') || name.includes('t1-ce')) {
        t1ceFile = f;
      } else if (name.includes('flair')) {
        flairFile = f;
      } else if (name.includes('t2')) {
        t2File = f;
      } else if (name.includes('t1')) {
        t1File = f;
      } else if (name.includes('seg') || name.includes('mask')) {
        segFile = f;
      } else {
        others.push(f);
      }
    });

    // If none matched specific modalities, fallback to first file
    const primaryFile = flairFile || t1ceFile || t1File || t2File || files[0];
    const primaryResult = await readNiftiRaw(primaryFile);

    window.AppState.setDatasetMetadata(primaryResult.meta);
    window.AppState.setVolumeData(primaryResult.voxels);

    const modalitiesMap = {};
    if (flairFile && flairFile === primaryFile) modalitiesMap['flair'] = primaryResult.voxels;
    if (t1File && t1File === primaryFile) modalitiesMap['t1'] = primaryResult.voxels;
    if (t1ceFile && t1ceFile === primaryFile) modalitiesMap['t1ce'] = primaryResult.voxels;
    if (t2File && t2File === primaryFile) modalitiesMap['t2'] = primaryResult.voxels;

    // Load remaining modalities asynchronously
    const otherModFiles = [
      { key: 't1', file: t1File },
      { key: 't1ce', file: t1ceFile },
      { key: 't2', file: t2File },
      { key: 'flair', file: flairFile }
    ].filter(item => item.file && item.file !== primaryFile);

    for (const item of otherModFiles) {
      try {
        const res = await readNiftiRaw(item.file);
        modalitiesMap[item.key] = res.voxels;
      } catch (e) {
        console.warn(`Could not parse modality ${item.key}:`, e);
      }
    }

    if (Object.keys(modalitiesMap).length > 0) {
      const activeKey = flairFile ? 'flair' : (t1ceFile ? 't1ce' : 't1');
      window.AppState.setModalities(modalitiesMap, activeKey);
    }

    // If BraTS segmentation mask is included, unpack and register it
    if (segFile) {
      try {
        const segRes = await readNiftiRaw(segFile);
        const segMask = new Uint8Array(segRes.voxels.length);
        for (let i = 0; i < segRes.voxels.length; i++) {
          segMask[i] = Math.round(segRes.voxels[i]);
        }
        window.AppState.setSegmentationMask(segMask, {
          1: 'Necrotic Core (NCR)',
          2: 'Peritumoral Edema (ED)',
          4: 'Enhancing Tumor (ET)'
        });
      } catch (e) {
        console.warn("Failed to parse BraTS _seg file:", e);
      }
    }

    return primaryResult.meta;
  }

  return {
    readNiftiRaw,
    parseNiftiFile,
    parseBraTSSeries
  };
})();
