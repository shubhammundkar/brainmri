/**
 * Standalone Browser NIfTI-1 Medical Image Parser
 * Conforms strictly to NIfTI-1 Header Field Specification Offsets.
 */

window.nifti = (function() {
  'use strict';

  const NIFTI_TYPE_UINT8 = 2;
  const NIFTI_TYPE_INT16 = 4;
  const NIFTI_TYPE_INT32 = 8;
  const NIFTI_TYPE_FLOAT32 = 16;
  const NIFTI_TYPE_FLOAT64 = 64;
  const NIFTI_TYPE_UINT16 = 512;
  const NIFTI_TYPE_INT8 = 256;
  const NIFTI_TYPE_UINT32 = 768;

  function isNIFTI(buffer) {
    if (!buffer || buffer.byteLength < 348) return false;
    const view = new DataView(buffer);
    const magic = String.fromCharCode(
      view.getUint8(344), view.getUint8(345), view.getUint8(346), view.getUint8(347)
    );
    return magic.substring(0, 3) === 'n+1' || magic.substring(0, 3) === 'ni1';
  }

  function readHeader(buffer) {
    const view = new DataView(buffer);
    let littleEndian = true;

    // Check endianness by dim[0] at offset 40
    let dim0 = view.getInt16(40, true);
    if (dim0 < 1 || dim0 > 7) {
      littleEndian = false;
      dim0 = view.getInt16(40, false);
    }

    const dims = [];
    for (let i = 0; i <= dim0; i++) {
      dims.push(view.getInt16(40 + i * 2, littleEndian));
    }

    const datatype = view.getInt16(70, littleEndian);
    const bitsPerVoxel = view.getInt16(72, littleEndian);

    const pixDims = [];
    for (let i = 0; i < 8; i++) {
      pixDims.push(view.getFloat32(80 + i * 4, littleEndian));
    }

    const voxOffset = view.getFloat32(108, littleEndian);
    let sclSlope = view.getFloat32(112, littleEndian);
    let sclInter = view.getFloat32(116, littleEndian);

    if (isNaN(sclSlope) || sclSlope === 0) sclSlope = 1;
    if (isNaN(sclInter)) sclInter = 0;

    const descriptionBytes = [];
    for (let i = 0; i < 80; i++) {
      const charCode = view.getUint8(148 + i);
      if (charCode === 0) break;
      descriptionBytes.push(String.fromCharCode(charCode));
    }
    const description = descriptionBytes.join('');

    return {
      littleEndian,
      dims,
      nx: dims[1] || 1,
      ny: dims[2] || 1,
      nz: dims[3] || 1,
      nt: dims[4] || 1,
      datatype,
      bitsPerVoxel,
      pixDims: [
        parseFloat((pixDims[1] || 1.0).toFixed(3)),
        parseFloat((pixDims[2] || 1.0).toFixed(3)),
        parseFloat((pixDims[3] || 1.0).toFixed(3))
      ],
      voxOffset: voxOffset || 352,
      sclSlope,
      sclInter,
      description
    };
  }

  function readImage(header, buffer) {
    const offset = Math.floor(header.voxOffset);
    const numVoxels = header.nx * header.ny * header.nz;
    const dataView = new DataView(buffer, offset);
    const voxels = new Float32Array(numVoxels);
    const isLE = header.littleEndian;

    let getVal;
    switch (header.datatype) {
      case NIFTI_TYPE_UINT8:
        getVal = (i) => dataView.getUint8(i);
        break;
      case NIFTI_TYPE_INT8:
        getVal = (i) => dataView.getInt8(i);
        break;
      case NIFTI_TYPE_INT16:
        getVal = (i) => dataView.getInt16(i * 2, isLE);
        break;
      case NIFTI_TYPE_UINT16:
        getVal = (i) => dataView.getUint16(i * 2, isLE);
        break;
      case NIFTI_TYPE_INT32:
        getVal = (i) => dataView.getInt32(i * 4, isLE);
        break;
      case NIFTI_TYPE_UINT32:
        getVal = (i) => dataView.getUint32(i * 4, isLE);
        break;
      case NIFTI_TYPE_FLOAT32:
        getVal = (i) => dataView.getFloat32(i * 4, isLE);
        break;
      case NIFTI_TYPE_FLOAT64:
        getVal = (i) => dataView.getFloat64(i * 8, isLE);
        break;
      default:
        getVal = (i) => dataView.getUint8(i);
    }

    const slope = header.sclSlope;
    const inter = header.sclInter;

    for (let i = 0; i < numVoxels; i++) {
      if (offset + i >= buffer.byteLength) break;
      const raw = getVal(i);
      voxels[i] = raw * slope + inter;
    }

    return voxels;
  }

  async function decompressIfNeeded(buffer, filename) {
    const uint8 = new Uint8Array(buffer);
    const isGzip = uint8[0] === 0x1f && uint8[1] === 0x8b;
    if (!isGzip && !filename.endsWith('.gz')) {
      return buffer;
    }

    // 1. Try pako if available
    if (window.pako && window.pako.ungzip) {
      try {
        const decompressed = window.pako.ungzip(uint8);
        return decompressed.buffer;
      } catch (e) {
        console.warn('Pako ungzip warning, attempting fallback:', e);
      }
    }

    // 2. Try native DecompressionStream
    if (typeof DecompressionStream !== 'undefined') {
      try {
        const ds = new DecompressionStream('gzip');
        const response = new Response(new Blob([buffer]).stream().pipeThrough(ds));
        return await response.arrayBuffer();
      } catch (err) {
        console.warn('DecompressionStream warning:', err);
      }
    }

    return buffer;
  }

  return {
    isNIFTI,
    readHeader,
    readImage,
    decompressIfNeeded
  };
})();
