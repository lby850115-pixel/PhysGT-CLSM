# Reproducibility

## Minimal Test

```bash
conda env create -f environment.yml
conda activate physgt-clsm
python src/physgt_clsm.py --img_dir data/test_images --out_pred outputs/predictions --out_fig outputs/figures --out_res outputs/results
```

Check that the following files are created:

- `outputs/predictions/S086_roi.tif`
- `outputs/figures/S086_r_physegt_clsm_overlay.png`
- `outputs/results/physegt_clsm_stats.csv`

## Synthetic Validation

```bash
python scripts/validate_synthetic.py
```

This regenerates synthetic CLSM-like image-label pairs and reports Dice, AJI and F1 at IoU 0.5.

## Full Manuscript Workflow

Place the full 34-image dataset in this structure:

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

## Notes

- The implementation is CPU-compatible.
- The repository does not include trained neural-network weights because PhysGT-CLSM does not require supervised training.
- Some comparison scripts expect baseline prediction folders from Cellpose, MitoSegNet, MoDL, Nellie and Mitometer.
