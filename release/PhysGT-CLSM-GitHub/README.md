# PhysGT-CLSM

Physics-informed annotation-free ground truth generation and mitochondria segmentation for confocal laser scanning microscopy (CLSM).

PhysGT-CLSM converts microscope and mitochondrial scale assumptions into a reproducible segmentation workflow. It simulates mitochondrial geometry, fluorophore placement, PSF blur, photon noise and readout noise, then applies adaptive thresholding and distance-transform watershed to produce instance masks without manual training labels.

## Repository Contents

```text
PhysGT-CLSM-GitHub/
├── src/
│   └── physgt_clsm.py
├── scripts/
│   ├── run_physgt_34.py
│   ├── validate_synthetic.py
│   ├── morphology_34.py
│   ├── stats_significance.py
│   └── paper_figures.py
├── data/
│   ├── test_images/
│   │   └── S086_roi.tif
│   └── test_labels/
│       └── S086_roi_gt.tif
├── figures/
│   └── S086_roi_reference.png
├── results/
│   ├── synthetic_validation_summary.txt
│   ├── roi_eval_summary.csv
│   └── stats_significance_table.csv
├── macros/
│   ├── fill_rois_as_labels.ijm
│   └── load_pred_rois.ijm
├── docs/
│   ├── DATA.md
│   ├── REPRODUCIBILITY.md
│   └── figure_index.md
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

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

## Reproduce Manuscript Analyses

The full 34-image analysis requires the complete microscopy dataset. The release package includes a minimal test image for code verification and compact result tables for manuscript traceability.

```bash
python scripts/validate_synthetic.py
python scripts/stats_significance.py
python scripts/paper_figures.py
```

For full-dataset runs, place the raw `.tif` files under:

```text
MITO DATA/
├── HELA/
├── BXPC-3/
└── MCF-7/
```

Then run:

```bash
python scripts/run_physgt_34.py
python scripts/morphology_34.py
python scripts/stats_significance.py
python scripts/paper_figures.py
```

## Main Physical Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| Pixel size | 120.25 nm/px | CLSM lateral calibration |
| Numerical aperture | 1.2 | Objective NA |
| Wavelength | 488 nm | Mitochondrial channel setting used in the implementation |
| PSF sigma | derived | `0.61 * wavelength / NA / 2.355 / pixel_size` |
| Minimum marker distance | about 10 px | Mitochondrial length-scale prior |
| Minimum object area | about 20 px2 | Diameter and minimum-length prior |

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
