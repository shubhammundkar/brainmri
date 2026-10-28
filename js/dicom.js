/**
 * BrainVision 3D - DICOM Series & Single File Parser Module (.dcm)
 */

window.DicomParserModule = (function() {
  'use strict';

  async function parseDicomFiles(files) {
    const fileList = Array.isArray(files) ? files : [files];
    if (fileList.length === 0) throw new Error('No DICOM files provided.');

    const parsedSlices = [];

    for (const file of fileList) {
      const buffer = await file.arrayBuffer();
      if (!window.dicomParserEngine.isDICOM(buffer)) {
        // Skip non-dicom files if in batch
        if (fileList.length > 1) continue;
        throw new Error('Invalid DICOM file format: Missing DICM signature at byte offset 128.');
      }

      const header = window.dicomParserEngine.parseDicomHeader(buffer);
      const pixels = window.dicomParserEngine.readDicomPixels(buffer, header);
      parsedSlices.push({ header, pixels, fileName: file.name });
    }

    if (parsedSlices.length === 0) {
      throw new Error('No valid DICOM files found in selection.');
    }

    // Sort slices by Instance Number
    parsedSlices.sort((a, b) => a.header.instanceNumber - b.header.instanceNumber);

    const first = parsedSlices[0];
    const nx = first.header.cols;
    const ny = first.header.rows;
    const nz = parsedSlices.length;
    const totalVoxels = nx * ny * nz;

    const volumeVoxels = new Float32Array(totalVoxels);
    const sliceVoxelCount = nx * ny;

    let minV = Infinity;
    let maxV = -Infinity;

    for (let z = 0; z < nz; z++) {
      const slicePixels = parsedSlices[z].pixels;
      const offset = z * sliceVoxelCount;
      for (let i = 0; i < sliceVoxelCount; i++) {
        const val = slicePixels[i] || 0;
        volumeVoxels[offset + i] = val;
        if (val < minV) minV = val;
        if (val > maxV) maxV = val;
      }
    }

    if (minV === Infinity) minV = 0;
    if (maxV === -Infinity) maxV = 255;

    const totalSizeMB = (fileList.reduce((acc, f) => acc + f.size, 0) / (1024 * 1024)).toFixed(2);

    const meta = {
      fileName: fileList.length === 1 ? first.fileName : `${fileList.length} DICOM Series Files`,
      fileSize: `${totalSizeMB} MB`,
      fileType: `DICOM Series (${nz} slice${nz > 1 ? 's' : ''})`,
      nx,
      ny,
      nz,
      dx: first.header.pixelSpacing[0] || 1.0,
      dy: first.header.pixelSpacing[1] || 1.0,
      dz: 1.0,
      datatype: `INT16 (${first.header.bitsAllocated}-bit)`,
      minVoxel: Math.round(minV),
      maxVoxel: Math.round(maxV)
    };

    window.AppState.setDatasetMetadata(meta);
    window.AppState.setVolumeData(volumeVoxels);
    return meta;
  }

  return {
    parseDicomFiles
  };
})();
