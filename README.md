# PhysGT-CLSM

Physics-informed annotation-free mitochondrial instance segmentation for confocal laser scanning microscopy (CLSM).

This repository contains a minimal public release of PhysGT-CLSM. The method generates mitochondrial instance labels directly from CLSM images using optical and morphological priors, without manual ground-truth annotation, neural-network training, or GPU-dependent inference.

## Repository Structure

```text
PhysGT-CLSM/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- PhysGT_CLSM.py
|-- validate_synthetic.py
|-- morphology_analysis.py
|-- example_data/
|   |-- raw_CLSM.tif
|   `-- segmentation_result.png
`-- figures/
    `-- workflow.png
```

## Method Overview

PhysGT-CLSM follows a deterministic physics-informed workflow:

1. PSF-aware foreground enhancement
2. Triangle-threshold foreground extraction
3. Distance-transform thickness mapping
4. Marker-controlled watershed separation
5. Small-object filtering and instance-label generation

The default real-image inference settings correspond to the optimized P3 preset used for the final 33-image CLSM analysis:

| Parameter | Value |
| --- | ---: |
| `smooth_sigma` | 1.0 |
| `threshold_scale` | 1.35 |
| `close_radius` | 0 |
| `dist_sigma` | 1.2 |
| `min_distance` | 8 px |
| `min_area` | 20 px |

## Dataset Summary

The full study used 33 real CLSM mitochondrial images from three human cancer cell lines. Raw full-size images are not bundled in this repository and should be distributed separately when a public data DOI is available.

Acquisition summary:

| Cell line | N images |
| --- | ---: |
| HeLa | 20 |
| BxPC-3 | 6 |
| MCF-7 | 7 |

Image metadata: 1024 x 1024 px, pixel size 120.25 nm/px, NA 1.2, excitation wavelength 488 nm.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Quick Start

Run PhysGT-CLSM on the bundled example CLSM tile:

```bash
python PhysGT_CLSM.py --img_dir example_data --out_pred predictions_33/physegt_clsm --out_fig figures/physegt_clsm --out_res results
```

Run synthetic self-validation:

```bash
python validate_synthetic.py
```

Run morphology analysis on available instance-label predictions:

```bash
python morphology_analysis.py
```

## Outputs

Typical outputs are generated locally and are ignored by Git:

```text
predictions_33/physegt_clsm/   uint16 instance-label TIF files
figures/physegt_clsm/          quick-look segmentation overlays
results/                       CSV summaries
```

## Notes

- This release intentionally contains only the code and minimal example materials required to reproduce the PhysGT-CLSM workflow.
- Draft documents, publication-specific figure scripts, baseline model weights, full raw datasets, and journal-specific materials are not included.
- The bundled example image is for workflow testing only.
