# Reproducibility

## Minimal Test

```bash
conda env create -f environment.yml
conda activate physgt-clsm
python src/physgt_clsm.py --img_dir data/test_images --out_pred outputs/predictions --out_fig outputs/figures --out_res outputs/results
```

The command above uses the final manuscript real-image working preset:

```text
smooth_sigma=1.0, threshold_scale=1.35, close_radius=0,
dist_sigma=1.2, min_distance=8, min_area=20
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

Place the full 33-image dataset in this structure:

```text
MITO DATA/
|-- HELA/
|-- BXPC-3/
`-- MCF-7/
```

Then run:

```bash
python scripts/run_physgt_34.py
python scripts/morphology_34.py
python scripts/stats_significance.py
python scripts/paper_figures.py
```

Note: some helper script names retain the earlier internal `_34` suffix, but the final manuscript statistics use the curated 33-image set.

## Notes

- The implementation is CPU-compatible.
- PhysGT-CLSM does not require supervised training or redistributed neural-network weights.
- Some comparison scripts expect baseline prediction folders from Cellpose, MitoSegNet, MoDL, Nellie and Mitometer.
