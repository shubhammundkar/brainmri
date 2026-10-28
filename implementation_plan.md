# Implementation Plan - BrainVision 3D (Phase 1 Frontend UI)

Build a high-precision, dark-mode medical imaging workstation UI for **BrainVision 3D** using standard HTML5, CSS3, and Vanilla JavaScript, adhering to Phase 1 constraints (frontend UI only, no backend/frameworks/libraries).

## User Review Required

> [!IMPORTANT]
> - Phase 1 strictly implements the **Visual Interface and Interactive Controls**.
> - No actual DICOM/NIfTI parsing, AI inference, or backend integration is performed in Phase 1.
> - HTML5 Canvas & SVG will be used to render realistic medical-style slice previews (axial, sagittal, coronal) and a 3D brain placeholder with responsive controls (contrast, brightness, opacity, window/level, zoom, maximize).

## Proposed Architecture & Directory Structure

Target structure as requested:
```
frontend/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── app.js
└── assets/
    └── (icons/placeholders if needed)
```

## Detailed Component Plan

### 1. `frontend/index.html`
- **Navigation Sidebar (Left)**
  - Brand identity: `BrainVision 3D` logo with medical icon & version badge.
  - Navigation tabs: `Dashboard`, `MRI Workspace`, `Analysis`, `Reports`, `Settings`.
  - Footer widgets: System Health Indicator (Green dot "Ready") & "Research Mode" status tag.
- **Main Workspace Header & Section Switcher**
  - Section 1: **Dashboard & MRI Upload Section**
    - Title: "Brain MRI Analysis"
    - Subtitle: "Upload an MRI dataset to begin 3D visualization and AI-assisted segmentation."
    - Drag & Drop Upload Zone:
      - Upload icon, "Upload Brain MRI", "Drag & drop your DICOM or NIfTI files here or [Browse Files]".
      - Supported formats info: `DICOM (.dcm)`, `NIfTI (.nii / .nii.gz)`.
      - Research & Privacy Disclaimer notice box.
      - Interactive file state banner ("Dataset selected: <filename>").
  - Section 2: **4-Panel Medical Viewport (3D Slicer / Raidionics layout)**
    - **Panel 1: AXIAL VIEW** (Top-Left)
    - **Panel 2: SAGITTAL VIEW** (Top-Right)
    - **Panel 3: CORONAL VIEW** (Bottom-Left)
    - **Panel 4: 3D VIEW** (Bottom-Right)
    - Each panel includes:
      - Header title & anatomical marker badges (A/P, L/R, S/I)
      - Compact panel toolbar: Zoom In/Out, Crosshair toggle, Reset, Maximize/Restore button
      - Viewport canvas element for procedural slice & 3D mesh rendering
      - 3D view placeholder overlay: "3D Reconstruction" / "Awaiting MRI processing"
- **Right Control & Analysis Panel**
  - **Visualization Control Panel**
    - Brain Visibility toggle checkbox / custom switch
    - Tumor Visibility toggle checkbox / custom switch
    - Tumor Opacity slider (0% - 100%)
    - MRI Opacity slider (0% - 100%)
    - Window / Level controls (Window Width & Window Level sliders + Presets: Soft Tissue, Bone, High Contrast)
    - Reset View button
  - **AI Analysis Panel**
    - Panel Title: "AI Analysis"
    - Metrics Card:
      - Status: "Awaiting MRI" / "Dataset Loaded"
      - Tumor Detected: `—`
      - Tumor Volume: `— cm³`
      - Tumor Region: `—`
      - Segmentation: `Not processed`
    - Action Button: `Run AI Segmentation` (Disabled appearance, non-functional per spec)
- **Bottom Status Bar**
  - System Status: `Ready`
  - MRI Dataset: `Not Loaded` / `<Filename>`
  - AI Model: `Not Loaded (Raidionics v2.1)`
  - 3D Engine: `Standby`

### 2. `frontend/css/style.css`
- **Design System Tokens**:
  - Backgrounds: `#090C10`, `#111622`, `#171E2E`, `#212B3E`
  - Accent colors: Medical Cyan `#00E5FF`, Active Blue `#2979FF`, Tumor Magenta `#FF2A6D`, Warning Amber `#FF9F1C`, Success Emerald `#00E676`
  - Typography: System sans-serif stack (`Inter`, `Segoe UI`, `system-ui`), crisp tabular numbers for metrics.
- **Layout Grid & Flexbox**:
  - Responsive desktop workstation layout with collapsible sidebar / flex containers.
  - 2x2 grid viewport layout with CSS Grid, supporting full-screen panel expansion (`.maximized` class).
  - Medical dark theme styling: 1px subtle borders (`rgba(255,255,255,0.08)`), glow highlights on active tools, custom sliders, crisp radio/switches.

### 3. `frontend/js/app.js`
- **UI State Management**:
  - `currentSection`: 'dashboard' | 'workspace' | 'analysis' | 'reports' | 'settings'
  - `datasetState`: `{ loaded: false, fileName: null, fileSize: null }`
  - `viewportState`: `{ brainVisible: true, tumorVisible: true, tumorOpacity: 0.8, mriOpacity: 1.0, windowWidth: 400, windowLevel: 40, zoomedPanel: null }`
- **Interactive Handlers**:
  - Navigation click listener for sidebar menu highlighting & view switching.
  - Drag-and-drop & file selection events (`dragover`, `dragleave`, `drop`, `change`):
    - Updates file upload dropzone visual states.
    - Captures file name and triggers "Dataset selected" notification banner.
    - Updates bottom status bar (`MRI Dataset: filename.nii.gz`) and Analysis status (`Dataset Selected`).
  - Medical Viewport Canvas Renderers:
    - Draw realistic slice representations (brain contours, white/gray matter texture gradients, ventricles, optional tumor highlight overlay) onto HTML5 Canvases for Axial, Sagittal, and Coronal views.
    - Re-render dynamically when MRI opacity, tumor opacity, brain visibility, tumor visibility, or Window/Level sliders are adjusted.
    - 3D Viewport: Render an interactive 3D styled wireframe/contour brain model with slow ambient rotation or static perspective display.
  - Viewport Panel Maximize / Restore toggle function.
  - Reset View button logic to restore default zoom/sliders/visibility settings.

## Verification Plan

### Automated / Browser Verification
1. Launch local web server (e.g. `python -m http.server 8000` or Node static server) serving the `frontend` folder.
2. Use browser subagent to test:
   - Sidebar tab navigation & active states.
   - Upload dropzone file selection & status bar update ("Dataset selected").
   - Panel maximize and restore functionality.
   - Control sliders (MRI Opacity, Tumor Opacity, Window/Level) and visibility toggles.
   - Disabled state of "Run AI Segmentation" button.
   - Responsiveness across different window sizes.

### Manual Verification Steps
- Ensure zero console errors in Developer Tools.
- Validate strict compliance with all constraints (no external CDN frameworks, no React/Tailwind, purely Vanilla HTML/CSS/JS).
