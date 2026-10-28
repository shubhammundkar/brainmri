/**
 * Unit Tests for PatientLoader (BraTS 2020 Multi-Modal Patient Ingestion)
 * Run with: node frontend/tests/patient_loader.test.js
 */

const assert = require('assert');
const PatientLoader = require('../js/patient-loader');

function testRoleDetection() {
  console.log('Testing detectFileRoleAndPatient...');

  // BraTS 2020 standard format
  const t1 = PatientLoader.detectFileRoleAndPatient('BraTS20_Training_369_t1.nii');
  assert.strictEqual(t1.role, 't1');
  assert.strictEqual(t1.patientId, 'BraTS20_Training_369');

  const t1ce = PatientLoader.detectFileRoleAndPatient('BraTS20_Training_369_t1ce.nii.gz');
  assert.strictEqual(t1ce.role, 't1ce');
  assert.strictEqual(t1ce.patientId, 'BraTS20_Training_369');

  const t2 = PatientLoader.detectFileRoleAndPatient('BraTS20_Training_369_t2.nii');
  assert.strictEqual(t2.role, 't2');
  assert.strictEqual(t2.patientId, 'BraTS20_Training_369');

  const flair = PatientLoader.detectFileRoleAndPatient('BraTS20_Training_369_flair.nii.gz');
  assert.strictEqual(flair.role, 'flair');
  assert.strictEqual(flair.patientId, 'BraTS20_Training_369');

  const seg = PatientLoader.detectFileRoleAndPatient('BraTS20_Training_369_seg.nii.gz');
  assert.strictEqual(seg.role, 'seg');
  assert.strictEqual(seg.patientId, 'BraTS20_Training_369');

  // Case-insensitive test
  const upper = PatientLoader.detectFileRoleAndPatient('PATIENT_001_FLAIR.NII.GZ');
  assert.strictEqual(upper.role, 'flair');
  assert.strictEqual(upper.patientId, 'PATIENT_001');

  // Non-NIfTI test
  const nonNifti = PatientLoader.detectFileRoleAndPatient('notes.txt');
  assert.strictEqual(nonNifti, null);

  console.log('  ✓ Role and patient detection tests passed.');
}

function testValidationCompleteSeries() {
  console.log('Testing validatePatientFiles with complete series...');

  const files = [
    { name: 'BraTS20_Training_369_t1.nii' },
    { name: 'BraTS20_Training_369_t1ce.nii' },
    { name: 'BraTS20_Training_369_t2.nii' },
    { name: 'BraTS20_Training_369_flair.nii' }
  ];

  const res = PatientLoader.validatePatientFiles(files);
  assert.strictEqual(res.valid, true);
  assert.strictEqual(res.patientId, 'BraTS20_Training_369');
  assert.ok(res.fileMap.t1);
  assert.ok(res.fileMap.t1ce);
  assert.ok(res.fileMap.t2);
  assert.ok(res.fileMap.flair);
  assert.strictEqual(res.fileMap.seg, null);

  console.log('  ✓ Complete 4-modality series validation passed.');
}

function testValidationWithGroundTruthSeg() {
  console.log('Testing validatePatientFiles with ground truth _seg...');

  const files = [
    { name: 'BraTS20_Training_369_t1.nii.gz' },
    { name: 'BraTS20_Training_369_t1ce.nii.gz' },
    { name: 'BraTS20_Training_369_t2.nii.gz' },
    { name: 'BraTS20_Training_369_flair.nii.gz' },
    { name: 'BraTS20_Training_369_seg.nii.gz' }
  ];

  const res = PatientLoader.validatePatientFiles(files);
  assert.strictEqual(res.valid, true);
  assert.strictEqual(res.patientId, 'BraTS20_Training_369');
  assert.ok(res.fileMap.seg);

  console.log('  ✓ 4 modalities + _seg ground truth passed.');
}

function testValidationMissingModality() {
  console.log('Testing validatePatientFiles missing modality...');

  // Missing t2 and flair
  const files = [
    { name: 'BraTS20_Training_369_t1.nii' },
    { name: 'BraTS20_Training_369_t1ce.nii' }
  ];

  const res = PatientLoader.validatePatientFiles(files);
  assert.strictEqual(res.valid, false);
  assert.ok(res.error.includes('Missing required modalities'));
  assert.ok(res.missingModalities.includes('t2'));
  assert.ok(res.missingModalities.includes('flair'));

  console.log('  ✓ Missing modalities properly detected and rejected.');
}

function testValidationPatientMismatch() {
  console.log('Testing validatePatientFiles mismatched patient IDs...');

  const files = [
    { name: 'BraTS20_Training_369_t1.nii' },
    { name: 'BraTS20_Training_369_t1ce.nii' },
    { name: 'BraTS20_Training_999_t2.nii' },
    { name: 'BraTS20_Training_369_flair.nii' }
  ];

  const res = PatientLoader.validatePatientFiles(files);
  assert.strictEqual(res.valid, false);
  assert.ok(res.error.includes('Patient ID Mismatch'));

  console.log('  ✓ Patient ID mismatch properly detected and rejected.');
}

function testVolumetricsCalculation() {
  console.log('Testing computeGroundTruthVolumetrics...');

  const nx = 10, ny = 10, nz = 10;
  const mask = new Uint8Array(nx * ny * nz);

  // 10 voxels of NCR (1), 20 of ED (2), 30 of ET (4)
  mask.fill(1, 0, 10);
  mask.fill(2, 10, 30);
  mask.fill(4, 30, 60);

  // Voxel spacing 1mm x 1mm x 1mm = 1 mm³ = 0.001 cm³
  const m = PatientLoader.computeGroundTruthVolumetrics(mask, nx, ny, nz, 1.0, 1.0, 1.0);
  assert.strictEqual(m.has_tumor, true);
  assert.strictEqual(m.total_tumor_cm3, 0.06); // 60 voxels = 0.06 cm³
  assert.strictEqual(m.enhancing_tumor_cm3, 0.03); // 30 voxels
  assert.strictEqual(m.edema_cm3, 0.02); // 20 voxels
  assert.strictEqual(m.necrotic_core_cm3, 0.01); // 10 voxels

  console.log('  ✓ Ground truth volumetrics verified.');
}

function runAllTests() {
  console.log('=== PatientLoader Unit Tests ===');
  testRoleDetection();
  testValidationCompleteSeries();
  testValidationWithGroundTruthSeg();
  testValidationMissingModality();
  testValidationPatientMismatch();
  testVolumetricsCalculation();
  console.log('=== All 6 Test Suites Passed Successfully! ===\n');
}

runAllTests();
