# Boundary Layer Extraction

This tool provides a modular and scriptable pipeline for exporting probe data and performing boundary layer spectral analysis using AVBP simulation output.

The main purpose of this tool is to: 
1. Extract the boundary layer profile 
2. Output WPS and BL parameters to the same format for WPS neural-network predictions

The input data are the time-series of surface probes, must be post-extracted though time-series extractions from raw-avbp data. 


It supports:
- Coordinate and pressure spectrum export from HDF5 probe files
- Full boundary layer profile extraction
- Nondimensional parameter calculation
- Modular CLI usage via `argparse`

---

## Project Structure

```
project_root/
│
├── extract_boundary_layer_core/
│   ├── __init__.py
│   ├── probe_exporter.py         # Contains ProbeDataExporter
│   ├── bl_extractor.py           # Contains BoundaryLayerExtractor
│
├── extract_boundary_layer.py                       # CLI entry point
```

---

## 🚀 How to Use

Run the tool from the terminal:

```bash
python main.py \
  --alpha 10 \
  --uref 30.0 \
  --rhoref 1.225 \
  --pref 101325.0 \
  --mu_lam 1.78e-5 \
  --chord 0.3048 \
  --input_dir ../../ \
  --mesh MESH_ZONE_Nov24/Bombardier_10AOA_Combine_Nov24.mesh.h5 \
  --solution PostProc/Average_Field/Averaged_Solution_Reduced_Variables.h5 \
  --probe_files Group_A_Probe_Data.h5 Group_B_Probe_Data.h5
```

---

## Features

### ProbeDataExporter
- Extracts probe coordinates and pressure signals
- Performs FFT analysis and saves pressure spectra as CSV
- Auto-creates output folders per AoA and probe group

### ✅ BoundaryLayerExtractor
- Loads AVBP mesh + mean solution
- Extracts and unwraps airfoil surface
- Separates suction/pressure sides
- Computes BL metrics: δ, θ, τw, Cf, Cp, Rt, PI, u_tau, etc.
- Writes results to structured CSV outputs

---

## Requirements

Python 3.8+ and the following packages:

- `numpy`
- `scipy`
- `h5py`
- `antares` (CFD-specific package)

---

## Module Imports

To use components in a script:

```python
from module import ProbeDataExporter, BoundaryLayerExtractor
```

---