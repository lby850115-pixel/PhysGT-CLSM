# Data

This release contains a minimal test dataset for verifying the PhysGT-CLSM code path.

## Included Files

| Path | Description |
| --- | --- |
| `data/test_images/S086_roi.tif` | CLSM mitochondria ROI used for a quick inference test |
| `data/test_labels/S086_roi_gt.tif` | ROI-level reference label map |
| `figures/S086_roi_reference.png` | Visual reference for the test ROI |
| `results/synthetic_validation_summary.txt` | Summary of synthetic self-validation metrics |
| `results/roi_eval_summary.csv` | ROI-level evaluation summary |
| `results/stats_significance_table.csv` | Pairwise statistical comparison summary |

## Full Dataset

The final manuscript evaluation used 33 real CLSM mitochondrial images from HeLa, BxPC-3 and MCF-7 cells. The complete raw dataset is not included in this compact GitHub release to avoid accidental redistribution of unpublished microscopy data.

Before publication, deposit the full dataset in an appropriate research data repository or provide an accession link in the manuscript Data Availability statement.

## Recommended Data Availability Statement

The source code, a minimal test dataset and processed result tables are available at the project GitHub repository. Full raw microscopy images supporting the manuscript findings will be deposited in a public repository before publication or made available from the corresponding author upon reasonable request.
