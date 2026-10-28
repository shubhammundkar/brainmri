/**
 * BrainVision 3D - Three.js WebGL Interactive 3D Medical Scene Module
 * Handles WebGL rendering, OrbitControls, physical spatial alignment,
 * brain & tumor isosurface reconstruction, opacity & layer controls, and memory cleanup.
 */

window.ThreeViewer = (function() {
  'use strict';

  let container = null;
  let canvas = null;
  let renderer = null;
  let scene = null;
  let camera = null;
  let controls = null;
  let animationFrameId = null;

  // Scene Objects
  let brainMesh = null;
  let tumorMesh = null;
  let tumorGroup = null;
  let boundingBoxHelper = null;

  // 3D Multi-Planar Slice Planes
  let slicePlaneAxial = null;
  let slicePlaneSagittal = null;
  let slicePlaneCoronal = null;
  let slicePlanesGroup = null;

  // 3D Clipping Planes
  let clipPlane = null;
  let isClippingEnabled = false;
  let currentClipAxis = 'y'; // 'y' = axial (superior/inferior), 'z' = coronal, 'x' = sagittal
  let currentClipValue = 0;

  // Materials
  let brainMaterial = null;
  let tumorMaterial = null;
  let multiclassMaterials = {};

  // Lights
  let ambientLight = null;
  let keyLight = null;
  let fillLight = null;
  let rimLight = null;

  // Viewer State
  let isInitialized = false;
  let isAutoRotating = false;
  let isReconstructing = false;
  let currentRenderMode = 'clinical'; // 'clinical', 'depth', 'glass'

  // Clinical Colors
  const COLORS = {
    brain: 0xd8e8f8,         // Professional warm white / translucent neuro-parenchyma
    tumorUnified: 0xef4444,  // Clinical focal red/coral
    et: 0xef4444,            // Enhancing Tumor: Red
    ed: 0xf59e0b,            // Peritumoral Edema: Amber / Yellow
    ncr: 0x06b6d4,           // Necrotic Core: Cyan
    background: 0x0b0f19     // Deep dark medical viewport backdrop
  };

  // Default camera configuration
  const DEFAULT_CAMERA = {
    fov: 42,
    pos: { x: 0, y: 30, z: 320 },
    target: { x: 0, y: 0, z: 0 }
  };

  /**
   * Initialize WebGL Scene, Camera, Lighting & OrbitControls
   */
  function init(containerElement, canvasElement) {
    if (!window.THREE) {
      console.error('Three.js library is not loaded.');
      showStatus('WebGL Engine Unavailable', 'Three.js library missing');
      return false;
    }

    container = containerElement || document.getElementById('screen-3d');
    canvas = canvasElement || document.getElementById('canvas-3d');

    if (!container || !canvas) {
      console.warn('3D Viewport DOM elements not found.');
      return false;
    }

    try {
      const width = container.clientWidth || 400;
      const height = container.clientHeight || 340;

      // 1. Scene
      scene = new THREE.Scene();
      scene.background = new THREE.Color(COLORS.background);

      // 2. Camera
      camera = new THREE.PerspectiveCamera(DEFAULT_CAMERA.fov, width / height, 0.1, 2000);
      camera.position.set(DEFAULT_CAMERA.pos.x, DEFAULT_CAMERA.pos.y, DEFAULT_CAMERA.pos.z);

      // 3. WebGL Renderer with Local Clipping
      renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.localClippingEnabled = true;

      // 4. OrbitControls
      if (window.THREE.OrbitControls) {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.rotateSpeed = 0.85;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.8;
        controls.target.set(DEFAULT_CAMERA.target.x, DEFAULT_CAMERA.target.y, DEFAULT_CAMERA.target.z);
        controls.maxDistance = 800;
        controls.minDistance = 30;
      }

      // 5. Clinical Lighting Rig
      setupLighting();

      // 6. Tumor Group & Slice Planes Group
      tumorGroup = new THREE.Group();
      scene.add(tumorGroup);

      slicePlanesGroup = new THREE.Group();
      scene.add(slicePlanesGroup);

      // 7. Clipping Plane Setup
      clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);

      isInitialized = true;

      // Start render loop
      startRenderLoop();

      // Hook resize observer
      const resizeObserver = new ResizeObserver(() => {
        handleResize();
      });
      resizeObserver.observe(container);

      showStatus('Awaiting MRI', 'Upload an MRI dataset to begin 3D reconstruction');
      return true;

    } catch (err) {
      console.error('Failed to initialize WebGL 3D Viewer:', err);
      showStatus('WebGL Error', 'Could not initialize 3D GPU context');
      return false;
    }
  }

  function setupLighting() {
    ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(100, 200, 200);
    scene.add(keyLight);

    fillLight = new THREE.DirectionalLight(0x90cdf4, 0.45);
    fillLight.position.set(-100, -100, -150);
    scene.add(fillLight);

    rimLight = new THREE.DirectionalLight(0x38bdf8, 0.4);
    rimLight.position.set(0, 150, -200);
    scene.add(rimLight);
  }

  function startRenderLoop() {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);

    function animate() {
      animationFrameId = requestAnimationFrame(animate);

      if (isAutoRotating && controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 2.0;
      } else if (controls) {
        controls.autoRotate = false;
      }

      if (controls) controls.update();
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
      }
    }
    animate();
  }

  /**
   * Handle dynamic container resizing
   */
  function handleResize() {
    if (!isInitialized || !container || !renderer || !camera) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  /**
   * Status UI Overlay helper
   */
  function showStatus(title, subtitle, isProgress = false) {
    const infoBox = container ? container.querySelector('.threed-reconstruction-info') : null;
    const tagStatus = document.getElementById('tag-3d-status');

    if (infoBox) {
      const titleEl = infoBox.querySelector('.threed-title');
      const subEl = infoBox.querySelector('.threed-subtitle');
      if (titleEl) titleEl.textContent = title;
      if (subEl) subEl.textContent = subtitle;
      infoBox.style.display = (title === 'Complete' || title === '3D Reconstruction Complete') ? 'none' : 'block';
    }

    if (tagStatus) {
      if (isProgress) {
        tagStatus.textContent = 'Processing...';
        tagStatus.className = 'pill-tag-muted processing';
      } else if (title.includes('Complete') || title.includes('Active') || title.includes('Ready')) {
        tagStatus.textContent = '3D Active ✓';
        tagStatus.className = 'pill-tag-muted success';
      } else {
        tagStatus.textContent = title;
        tagStatus.className = 'pill-tag-muted';
      }
    }
  }

  /**
   * Reset 3D Camera & Target to default without disturbing 2D views
   */
  function reset3DView() {
    if (!camera || !controls) return;
    camera.position.set(DEFAULT_CAMERA.pos.x, DEFAULT_CAMERA.pos.y, DEFAULT_CAMERA.pos.z);
    controls.target.set(DEFAULT_CAMERA.target.x, DEFAULT_CAMERA.target.y, DEFAULT_CAMERA.target.z);
    controls.update();
  }

  /**
   * Toggle auto-rotation
   */
  function toggleAutoRotation() {
    isAutoRotating = !isAutoRotating;
    return isAutoRotating;
  }

  /**
   * Dispose and clear existing 3D geometries and materials
   */
  function disposeMeshes() {
    if (brainMesh) {
      scene.remove(brainMesh);
      if (brainMesh.geometry) brainMesh.geometry.dispose();
      if (brainMesh.material) brainMesh.material.dispose();
      brainMesh = null;
    }

    if (tumorGroup) {
      while (tumorGroup.children.length > 0) {
        const child = tumorGroup.children[0];
        tumorGroup.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      }
    }

    if (boundingBoxHelper) {
      scene.remove(boundingBoxHelper);
      if (boundingBoxHelper.geometry) boundingBoxHelper.geometry.dispose();
      if (boundingBoxHelper.material) boundingBoxHelper.material.dispose();
      boundingBoxHelper = null;
    }

    tumorMesh = null;
    multiclassMaterials = {};
  }

  /**
   * Asynchronously reconstruct Brain Surface from real MRI Volume
   */
  async function reconstructBrainSurface(onProgress) {
    const state = window.AppState.state;
    const { dataset, volume } = state;

    if (!volume || dataset.nx === 0) {
      showStatus('Awaiting MRI', 'Upload an MRI dataset to begin 3D reconstruction');
      return;
    }

    if (!window.MarchingCubes) {
      console.error('MarchingCubes algorithm module missing.');
      return;
    }

    isReconstructing = true;
    showStatus('Preparing Volume...', 'Analyzing voxel intensity distribution', true);
    if (onProgress) onProgress('Preparing volume...');

    await new Promise(r => setTimeout(r, 40));

    try {
      showStatus('Extracting Brain Surface...', 'Running 3D Marching Cubes algorithm', true);
      if (onProgress) onProgress('Extracting brain surface...');
      await new Promise(r => setTimeout(r, 40));

      const { nx, ny, nz, dx, dy, dz, minVoxel, maxVoxel } = dataset;
      
      // Auto-compute parenchyma threshold (approx 22% of intensity range above background)
      const range = Math.max(1, maxVoxel - minVoxel);
      const isovalue = minVoxel + range * 0.22;

      // Stride of 2 gives high resolution (< 100ms extraction) while preserving cortical gyri
      const step = 2;
      const vertices = window.MarchingCubes.extractIsosurface(
        volume, nx, ny, nz, dx, dy, dz, isovalue, step
      );

      if (!vertices || vertices.length === 0) {
        showStatus('Reconstruction Notice', 'Low intensity gradient detected in volume');
        isReconstructing = false;
        return;
      }

      showStatus('Building 3D Geometry...', 'Compiling indexed WebGL buffer attributes', true);
      if (onProgress) onProgress('Building 3D geometry...');
      await new Promise(r => setTimeout(r, 30));

      // Dispose existing brain mesh
      if (brainMesh) {
        scene.remove(brainMesh);
        if (brainMesh.geometry) brainMesh.geometry.dispose();
        if (brainMesh.material) brainMesh.material.dispose();
        brainMesh = null;
      }

      // Create BufferGeometry
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      geometry.computeVertexNormals();

      // Translucent clinical brain parenchyma material
      const brainOpacity = state.viewport.brainOpacity !== undefined ? state.viewport.brainOpacity : 0.35;
      brainMaterial = new THREE.MeshStandardMaterial({
        color: COLORS.brain,
        roughness: 0.45,
        metalness: 0.1,
        transparent: true,
        opacity: brainOpacity,
        depthWrite: false, // Ensure interior tumor is clearly visible through translucent shell
        side: THREE.DoubleSide
      });

      brainMesh = new THREE.Mesh(geometry, brainMaterial);
      brainMesh.visible = state.viewport.brain3DVisible !== undefined ? state.viewport.brain3DVisible : true;
      scene.add(brainMesh);

      // Create Subtle Bounding Box Guide
      createBoundingBoxGuide(nx * dx, nz * dz, ny * dy);

      // Adjust camera distance to fit brain model
      const maxDim = Math.max(nx * dx, ny * dy, nz * dz);
      camera.position.set(0, maxDim * 0.2, maxDim * 1.65);
      if (controls) {
        controls.target.set(0, 0, 0);
        controls.update();
      }

      isReconstructing = false;
      showStatus('3D Reconstruction Complete', 'Brain surface generated from real MRI anatomy');
      if (onProgress) onProgress('Complete');

    } catch (err) {
      console.error('Brain 3D reconstruction failed:', err);
      showStatus('Reconstruction Error', err.message);
      isReconstructing = false;
    }
  }

  /**
   * Asynchronously reconstruct Tumor Surface from real Phase 3 Segmentation Mask
   */
  async function reconstructTumorSurface(onProgress) {
    const state = window.AppState.state;
    const { dataset, segmentationMask, segmentationLabels } = state;

    if (!segmentationMask || segmentationMask.length === 0 || dataset.nx === 0) {
      // Clear tumor meshes if mask is removed
      if (tumorGroup) {
        while (tumorGroup.children.length > 0) {
          const c = tumorGroup.children[0];
          tumorGroup.remove(c);
          if (c.geometry) c.geometry.dispose();
          if (c.material) c.material.dispose();
        }
      }
      return;
    }

    if (!window.MarchingCubes) return;

    try {
      showStatus('Generating Tumor Surface...', 'Extracting 3D lesion boundary isosurface', true);
      if (onProgress) onProgress('Generating tumor surface...');
      await new Promise(r => setTimeout(r, 40));

      const { nx, ny, nz, dx, dy, dz } = dataset;

      // Clear existing tumor children
      while (tumorGroup.children.length > 0) {
        const c = tumorGroup.children[0];
        tumorGroup.remove(c);
        if (c.geometry) c.geometry.dispose();
        if (c.material) c.material.dispose();
      }

      const tumorOpacity = state.viewport.tumorOpacity !== undefined ? state.viewport.tumorOpacity : 0.95;
      const tumorVisible = state.viewport.tumor3DVisible !== undefined ? state.viewport.tumor3DVisible : true;

      // Check if multi-class labels exist or unified tumor
      const labels = segmentationLabels && Object.keys(segmentationLabels).length > 0
        ? Object.keys(segmentationLabels).map(k => parseInt(k, 10))
        : [1];

      // Extract each label as an independent spatially aligned mesh
      for (const labelId of labels) {
        const targetLabel = [labelId];
        const vertices = window.MarchingCubes.extractMaskSurface(
          segmentationMask, nx, ny, nz, dx, dy, dz, targetLabel, 1
        );

        if (vertices && vertices.length > 0) {
          const geom = new THREE.BufferGeometry();
          geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
          geom.computeVertexNormals();

          // Clinical Color Mapping per label
          let matColor = COLORS.tumorUnified;
          if (labelId === 1) matColor = COLORS.ncr;      // Necrotic Core (Cyan)
          else if (labelId === 2) matColor = COLORS.ed;  // Edema (Amber)
          else if (labelId === 3 || labelId === 4) matColor = COLORS.et;  // Enhancing Tumor (Red)

          const mat = new THREE.MeshStandardMaterial({
            color: matColor,
            roughness: 0.3,
            metalness: 0.2,
            transparent: tumorOpacity < 1.0,
            opacity: tumorOpacity,
            side: THREE.DoubleSide,
            depthTest: true
          });

          const mesh = new THREE.Mesh(geom, mat);
          mesh.userData = { labelId };
          tumorGroup.add(mesh);
          multiclassMaterials[labelId] = mat;
        }
      }

      // If multiclass extraction returned empty, try composite all nonzero labels
      if (tumorGroup.children.length === 0) {
        const vertices = window.MarchingCubes.extractMaskSurface(
          segmentationMask, nx, ny, nz, dx, dy, dz, [1, 2, 3, 4], 1
        );
        if (vertices && vertices.length > 0) {
          const geom = new THREE.BufferGeometry();
          geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
          geom.computeVertexNormals();

          tumorMaterial = new THREE.MeshStandardMaterial({
            color: COLORS.tumorUnified,
            roughness: 0.35,
            metalness: 0.2,
            transparent: tumorOpacity < 1.0,
            opacity: tumorOpacity,
            side: THREE.DoubleSide
          });

          const mesh = new THREE.Mesh(geom, tumorMaterial);
          mesh.userData = { labelId: 0 };
          tumorGroup.add(mesh);
        }
      }

      tumorGroup.visible = tumorVisible;

      showStatus('3D Reconstruction Complete', 'Brain and tumor surfaces fully aligned');
      if (onProgress) onProgress('Complete');

    } catch (err) {
      console.error('Tumor 3D reconstruction error:', err);
      showStatus('Tumor 3D Error', err.message);
    }
  }

  function createBoundingBoxGuide(sx, sy, sz) {
    if (boundingBoxHelper) {
      scene.remove(boundingBoxHelper);
      if (boundingBoxHelper.geometry) boundingBoxHelper.geometry.dispose();
      boundingBoxHelper = null;
    }

    const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
    const wireGeom = new THREE.WireframeGeometry(boxGeom);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.18
    });

    boundingBoxHelper = new THREE.LineSegments(wireGeom, lineMat);
    scene.add(boundingBoxHelper);

    // Also initialize 3D Multi-Planar Slice bounding planes
    init3DSlicePlanes(sx, sy, sz);
  }

  /**
   * Initialize 3D Multi-Planar Slice Reference Rectangles
   */
  function init3DSlicePlanes(sx, sy, sz) {
    if (!slicePlanesGroup) return;

    while (slicePlanesGroup.children.length > 0) {
      const child = slicePlanesGroup.children[0];
      slicePlanesGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    // 1. Axial Slice Plane (XZ plane, green/cyan)
    const axialGeom = new THREE.PlaneGeometry(sx, sz);
    const axialMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    slicePlaneAxial = new THREE.Mesh(axialGeom, axialMat);
    slicePlaneAxial.rotation.x = Math.PI / 2;
    slicePlanesGroup.add(slicePlaneAxial);

    // 2. Sagittal Slice Plane (YZ plane, neon emerald)
    const sagGeom = new THREE.PlaneGeometry(sz, sy);
    const sagMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    slicePlaneSagittal = new THREE.Mesh(sagGeom, sagMat);
    slicePlaneSagittal.rotation.y = Math.PI / 2;
    slicePlanesGroup.add(slicePlaneSagittal);

    // 3. Coronal Slice Plane (XY plane, amber)
    const corGeom = new THREE.PlaneGeometry(sx, sy);
    const corMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    slicePlaneCoronal = new THREE.Mesh(corGeom, corMat);
    slicePlanesGroup.add(slicePlaneCoronal);

    slicePlanesGroup.visible = true;
    update3DSlicePlanes();
  }

  /**
   * Synchronize 3D slice planes with 2D slice scrubber positions
   */
  function update3DSlicePlanes() {
    const state = window.AppState.state;
    const { dataset, viewport } = state;
    if (!dataset || dataset.nx === 0) return;

    const { nx, ny, nz, dx, dy, dz } = dataset;
    const { slice } = viewport;

    if (slicePlaneAxial) {
      // Axial moves along Y-axis in 3D (z in NIfTI)
      const curZ = slice.axial !== undefined ? slice.axial : nz / 2;
      slicePlaneAxial.position.y = (curZ - nz / 2) * dz;
    }

    if (slicePlaneSagittal) {
      // Sagittal moves along X-axis in 3D (x in NIfTI)
      const curX = slice.sagittal !== undefined ? slice.sagittal : nx / 2;
      slicePlaneSagittal.position.x = (curX - nx / 2) * dx;
    }

    if (slicePlaneCoronal) {
      // Coronal moves along Z-axis in 3D (y in NIfTI)
      const curY = slice.coronal !== undefined ? slice.coronal : ny / 2;
      slicePlaneCoronal.position.z = (curY - ny / 2) * dy;
    }
  }

  function setSlicePlanesVisible(visible) {
    if (slicePlanesGroup) slicePlanesGroup.visible = !!visible;
  }

  /**
   * 3D Cross-Sectional Cut / Slicing Plane
   */
  function setClipping(enabled, axis = 'y', offsetNorm = 0.5) {
    isClippingEnabled = !!enabled;
    currentClipAxis = axis;

    const state = window.AppState.state;
    const { dataset } = state;
    if (!dataset || dataset.nx === 0) return;

    const { nx, ny, nz, dx, dy, dz } = dataset;

    let normal = new THREE.Vector3(0, -1, 0); // Axial cut default
    let maxExtent = (nz * dz) / 2;

    if (axis === 'x') { // Sagittal cut
      normal = new THREE.Vector3(-1, 0, 0);
      maxExtent = (nx * dx) / 2;
    } else if (axis === 'z') { // Coronal cut
      normal = new THREE.Vector3(0, 0, -1);
      maxExtent = (ny * dy) / 2;
    }

    const constant = (offsetNorm - 0.5) * 2 * maxExtent;
    clipPlane.set(normal, constant);

    if (brainMaterial) {
      brainMaterial.clippingPlanes = isClippingEnabled ? [clipPlane] : [];
      brainMaterial.clipShadows = true;
      brainMaterial.needsUpdate = true;
    }
  }

  /**
   * Quick-View Preset Angles (Anterior, Posterior, Superior, Lateral)
   */
  function setQuickView(viewName) {
    if (!camera || !controls) return;
    const state = window.AppState.state;
    const { dataset } = state;
    const maxDim = dataset && dataset.nx > 0
      ? Math.max(dataset.nx * dataset.dx, dataset.ny * dataset.dy, dataset.nz * dataset.dz)
      : 220;
    const dist = maxDim * 1.65;

    switch (viewName.toLowerCase()) {
      case 'anterior':
      case 'front':
        camera.position.set(0, 0, dist);
        break;
      case 'posterior':
      case 'back':
        camera.position.set(0, 0, -dist);
        break;
      case 'superior':
      case 'top':
        camera.position.set(0, dist, 0.001);
        break;
      case 'inferior':
      case 'bottom':
        camera.position.set(0, -dist, 0.001);
        break;
      case 'left':
        camera.position.set(dist, 0, 0);
        break;
      case 'right':
        camera.position.set(-dist, 0, 0);
        break;
      case 'isometric':
      default:
        camera.position.set(dist * 0.7, dist * 0.5, dist * 0.7);
        break;
    }
    controls.target.set(0, 0, 0);
    controls.update();
  }

  /**
   * 3D Lighting & Rendering Presets
   */
  function setRenderMode(mode) {
    currentRenderMode = mode;
    if (!brainMaterial) return;

    if (mode === 'depth') {
      // Cortical Depth Contrast (highlights sulci/gyri fissures)
      brainMaterial.roughness = 0.75;
      brainMaterial.metalness = 0.05;
      if (ambientLight) ambientLight.intensity = 0.45;
      if (keyLight) keyLight.intensity = 1.1;
      if (rimLight) rimLight.intensity = 0.7;
    } else if (mode === 'glass') {
      // Translucent Glassmorphism (maximum interior tumor visibility)
      brainMaterial.roughness = 0.15;
      brainMaterial.metalness = 0.35;
      brainMaterial.opacity = 0.22;
      brainMaterial.transparent = true;
      if (ambientLight) ambientLight.intensity = 0.8;
      if (keyLight) keyLight.intensity = 0.9;
    } else {
      // Clinical Radiographic Default
      brainMaterial.roughness = 0.45;
      brainMaterial.metalness = 0.1;
      const baseOp = window.AppState.state.viewport.brainOpacity || 0.35;
      brainMaterial.opacity = baseOp;
      if (ambientLight) ambientLight.intensity = 0.65;
      if (keyLight) keyLight.intensity = 0.85;
      if (rimLight) rimLight.intensity = 0.4;
    }
    brainMaterial.needsUpdate = true;
  }

  // -------------------------------------------------------------
  // Controls & Property Setters
  // -------------------------------------------------------------
  function setBrainVisibility(visible) {
    if (brainMesh) brainMesh.visible = !!visible;
    window.AppState.state.viewport.brain3DVisible = !!visible;
  }

  function setTumorVisibility(visible) {
    if (tumorGroup) tumorGroup.visible = !!visible;
    window.AppState.state.viewport.tumor3DVisible = !!visible;
  }

  function setBrainOpacity(opacity) {
    const op = Math.max(0.05, Math.min(1.0, parseFloat(opacity)));
    if (brainMaterial) {
      brainMaterial.opacity = op;
      brainMaterial.transparent = op < 1.0;
    }
    window.AppState.state.viewport.brainOpacity = op;
  }

  function setTumorOpacity(opacity) {
    const op = Math.max(0.1, Math.min(1.0, parseFloat(opacity)));
    if (tumorGroup) {
      tumorGroup.children.forEach(mesh => {
        if (mesh.material) {
          mesh.material.opacity = op;
          mesh.material.transparent = op < 1.0;
        }
      });
    }
    window.AppState.state.viewport.tumorOpacity = op;
  }

  function setComponentVisibility(labelId, visible) {
    if (tumorGroup) {
      tumorGroup.children.forEach(mesh => {
        if (mesh.userData && mesh.userData.labelId !== undefined) {
          const mLabel = mesh.userData.labelId;
          const match = (mLabel === labelId) ||
                        ((labelId === 3 || labelId === 4) && (mLabel === 3 || mLabel === 4));
          if (match) {
            mesh.visible = !!visible;
          }
        }
      });
    }
  }

  return {
    init,
    reconstructBrainSurface,
    reconstructTumorSurface,
    setBrainVisibility,
    setTumorVisibility,
    setBrainOpacity,
    setTumorOpacity,
    setComponentVisibility,
    setClipping,
    update3DSlicePlanes,
    setSlicePlanesVisible,
    setQuickView,
    setRenderMode,
    reset3DView,
    toggleAutoRotation,
    handleResize,
    disposeMeshes,
    showStatus
  };
})();
