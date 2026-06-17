# PhysGT-CLSM

Physics-informed annotation-free mitochondrial instance generation and morphology analysis for confocal laser scanning microscopy (CLSM).

PhysGT-CLSM converts microscope-scale optical priors and mitochondrial morphology assumptions into a reproducible segmentation workflow. It combines synthetic CLSM image generation, physics-derived reference labels, foreground detection, distance-transform analysis and marker-controlled watershed segmentation to generate mitochondrial instance candidates without manual training labels or neural-network training.

## What This Repository Contains

```text
PhysGT-CLSM-GitHub/
|-- src/
|   `-- physgt_clsm.py
|-- scripts/
|   |-- run_physgt_34.py
|   |-- validate_synthetic.py
|   |-- morphology_34.py
|   |-- stats_significance.py
|   `-- paper_figures.py
|-- baselines/
|   |-- README.md
|   |-- baseline_versions.yml
|   |-- run_cellpose.py
|   |-- run_mitosegnet_34.py
|   |-- run_modl_34.py
|   |-- run_nellie_style.py
|   `-- run_mitometer_style.py
|-- data/
|   |-- test_images/
|   |   `-- S086_roi.tif
|   `-- test_labels/
|       `-- S086_roi_gt.tif
|-- figures/
|-- results/
|-- outputs/
|-- macros/
|-- docs/
|-- environment.yml
|-- requirements.txt
|-- CITATION.cff
`-- LICENSE
```

Some helper script names retain the earlier internal suffix `_34`; the final manuscript statistics reported in the paper use 33 CLSM images.

## Installation

```bash
conda env create -f environment.yml
conda activate physgt-clsm
```

Alternatively:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Test

Run the included ROI-level test image:

```bash
python src/physgt_clsm.py --img_dir data/test_images --out_pred outputs/predictions --out_fig outputs/figures --out_res outputs/results
```

Expected outputs:

- `outputs/predictions/*.tif`: uint16 instance-label maps
- `outputs/figures/*_physegt_clsm_overlay.png`: quality-control overlays
- `outputs/results/physegt_clsm_stats.csv`: per-image summary

For the included `S086_roi.tif` test image, the current manuscript working preset detects 28 instances with median area 422 px in the reference environment. A representative output is included at `figures/S086_physgt_working_preset_overlay.png`. Minor differences can occur across dependency versions.

## Manuscript Working Preset

The final real-image analysis used a morphology-preserving working preset selected from the parameter-sensitivity curve:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--smooth_sigma` | 1.0 | Gaussian pre-smoothing before thresholding |
| `--threshold_scale` | 1.35 | Multiplier applied to the Triangle threshold |
| `--close_radius` | 0 | Binary closing radius; `0` disables closing |
| `--dist_sigma` | 1.2 | Distance-transform smoothing before marker detection |
| `--min_distance` | 8 | Minimum watershed marker spacing in pixels |
| `--min_area` | 20 | Minimum retained instance area in pixels |

This preset is intended to balance excessive fragmentation and under-separation in dense ring-like or network-like mitochondrial regions. It is not optimized against manual real-image ground truth.

## Main Physical Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| Pixel size | 120.25 nm/px | CLSM lateral calibration |
| Numerical aperture | 1.2 | Objective NA |
| Wavelength | 488 nm | Mitochondrial channel setting used in the implementation |
| PSF sigma | derived | `0.61 * wavelength / NA / 2.355 / pixel_size` |

## Reproduce Manuscript Analyses

The compact GitHub release includes a minimal test image, processed reference outputs and summary result tables. The full raw microscopy dataset is not redistributed in this repository.

```bash
python scripts/validate_synthetic.py
python scripts/stats_significance.py
python scripts/paper_figures.py
```

For full-dataset runs, place raw `.tif` files under:

```text
MITO DATA/
|-- HELA/
|-- BXPC-3/
`-- MCF-7/
```

Then run the full analysis helpers:

```bash
python scripts/run_physgt_34.py
python scripts/morphology_34.py
python scripts/stats_significance.py
python scripts/paper_figures.py
```

Baseline comparison wrappers are documented in `baselines/`. Third-party source code, plugins, pretrained weights and full raw microscopy data are not redistributed; install them from official sources and use the wrappers to reproduce the manuscript comparison outputs.

## Synthetic Self-Validation Summary

The manuscript synthetic self-validation used 100 independently generated synthetic CLSM tiles:

```text
Dice (semantic)   : 0.847 +/- 0.041
AJI               : 0.650 +/- 0.106
F1 @ IoU=0.5      : 0.740 +/- 0.149
GT count / tile   : 9.8 +/- 2.3
Pred count / tile : 8.3 +/- 2.0
```

## Citation

If you use this repository, please cite:

```bibtex
@article{physgt_clsm,
  title = {PhysGT-CLSM: a physics-informed framework for annotation-free ground truth generation and mitochondria segmentation in confocal microscopy},
  author = {Author Names},
  journal = {Computer Methods and Programs in Biomedicine},
  year = {2026},
  note = {Manuscript in preparation}
}
```

## License

This project is distributed under the MIT License.
