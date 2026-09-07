# PhysGT-CLSM

PhysGT-CLSM is a lightweight, physics-informed pipeline for annotation-free mitochondrial instance segmentation in confocal laser scanning microscopy (CLSM). It uses no neural-network training, model weights, or GPU.

## Manuscript release

This branch reproduces the protocol used in the manuscript's curated 33-image analysis.

- Dataset: 33 real CLSM images
- Cell lines: HeLa (20), BxPC-3 (5), and MCF-7 (8)
- Image size: 1024 x 1024 pixels
- Lateral sampling: 120.25 nm/pixel
- Numerical aperture: 1.2
- Wavelength used for PSF calibration: 488 nm

The exact image list is frozen in `configs/manuscript_33.csv`. Full images and third-party baseline predictions are not redistributed in this repository; see `DATA.md`.

## Real-image P3 protocol

The public real-image command implements the same P3 pipeline used for the manuscript figures and morphology analysis:

1. Robust percentile normalization (1st-99th percentiles).
2. Gaussian PSF approximation with sigma = 0.876 pixels.
3. PSF-informed unsharp compensation with gain 0.8.
4. Otsu foreground thresholding.
5. Binary closing with a one-pixel disk.
6. Distance-transform smoothing with sigma = 1.2 pixels.
7. Marker-controlled watershed with minimum marker distance = 8 pixels.
8. Removal of instances smaller than 20 pixels.

The P3 settings are fixed defaults. The synthetic self-validation uses a separate simulation-domain recovery protocol and is not presented as an identical real-image thresholding procedure.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For the exact tested environment, use `environment.yml`.

## Quick start

Run the bundled example:

```bash
python PhysGT_CLSM.py
```

Expected result:

- 29 instances
- median instance area: 207 pixels
- mean instance area: 266 pixels

Run a folder recursively:

```bash
python PhysGT_CLSM.py --img_dir /path/to/images \
  --out_pred predictions_33/physegt_clsm \
  --out_fig figures/physegt_clsm \
  --out_res results
```

Run deterministic synthetic self-validation:

```bash
python validate_synthetic.py
```

Expected results for 100 tiles with seed 42:

| Metric | Mean | SD |
| --- | ---: | ---: |
| Dice | 0.847 | 0.041 |
| AJI | 0.650 | 0.106 |
| F1 at IoU 0.5 | 0.740 | 0.149 |

Run the lightweight regression test:

```bash
python tests/smoke_test.py
```

## Morphology analysis

The morphology script requires the 33-image manifest and one prediction subdirectory per model:

```text
predictions_33/
  physegt_clsm/
  cellpose/
  mitosegnet/
  modl/
  nellie/
  mitometer/
```

```bash
python morphology_analysis.py \
  --manifest configs/manuscript_33.csv \
  --pred-dir predictions_33
```

Only models with available prediction directories are processed. Reproducing the complete six-model table requires the third-party predictions described in `DATA.md`.

## Outputs

- `predictions_33/physegt_clsm/*.tif`: uint16 instance-label images
- `figures/physegt_clsm/*.png`: QC overlays
- `results/physegt_clsm_stats.csv`: per-image counts and areas
- `results/synthetic_validation.csv`: per-tile validation metrics
- `results/morphology_33/`: morphology tables

## Reproducibility scope

The bundled example and synthetic validation are fully self-contained. The complete 33-image and six-model analyses require controlled-access raw data and baseline outputs. The repository records the exact cohort, parameters, software environment, and expected smoke-test values so those analyses can be rerun when the data package is supplied.

## License

MIT License. See `LICENSE`.
