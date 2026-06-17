# Baseline Model Reproduction

This directory documents how the five comparison methods in the manuscript were run. It contains only lightweight wrapper or re-implementation scripts used for evaluation. It does not redistribute third-party model repositories, pretrained weights, plugins, or full raw microscopy datasets.

## Included Baseline Wrappers

| Method | File | Redistribution note |
| --- | --- | --- |
| Cellpose | `run_cellpose.py` | Uses the official `cellpose` Python package installed by the user. No Cellpose source code or weights are bundled. |
| MitoSegNet | `run_mitosegnet_34.py` | Uses a U-Net style inference wrapper and expects a local checkpoint path. The checkpoint is not bundled. |
| MoDL | `run_modl_34.py` | Expects the original MoDL/U-RNet+ weight file at a local path. The weight file is not bundled. |
| Nellie-style | `run_nellie_style.py` | Reproduces the analysis behavior used in this study with multiscale Frangi filtering and connected components. |
| Mitometer-style | `run_mitometer_style.py` | Reproduces the segmentation behavior used in this study with diffuse background subtraction, parameter search and connected components. |

## Expected Full-Dataset Layout

The final manuscript comparison used the curated 33-image CLSM dataset. Server-side runs used the following project layout:

```text
/home/zhishi/LBY_CELL/
|-- data/
|   `-- mito_33/
|       |-- HELA/
|       |-- BXPC3/
|       `-- MCF7/
|-- checkpoints/
|   `-- mitosegnet/
|       `-- mitosegnet_best.pt
|-- MoDL/
|   `-- model/
|       `-- U-RNet+.hdf5
|-- results/
`-- figures/
```

The local single-directory scripts expect:

```text
MITO DATA/
```

as the input folder beside the script.

Note: several wrapper filenames retain an earlier internal `_34` suffix. This suffix is historical and does not change the final 33-image manuscript analysis.

## Commands Used for Comparison

Cellpose:

```bash
python baselines/run_cellpose.py --model bact_fluor_cp3 --diameter 0 --flow_thr 0.4 --prob_thr 0.0 --min_area 5
```

MitoSegNet:

```bash
python baselines/run_mitosegnet_34.py
```

MoDL:

```bash
python baselines/run_modl_34.py
```

Nellie-style:

```bash
python baselines/run_nellie_style.py --min_area 25
```

Mitometer-style:

```bash
python baselines/run_mitometer_style.py --channel 1 --min_area 5 --max_area 2000
```

## Output Convention

Each wrapper writes:

- instance label maps as `.tif`
- quality-control overlays as `.png`
- per-image statistics as `.csv`

Downstream morphology and statistical comparison are performed by:

```bash
python scripts/morphology_34.py
python scripts/stats_significance.py
```

## Citation and License Boundary

Please install third-party tools from their official sources and cite their original publications. This repository provides the reproducibility glue used for manuscript comparison, not redistributed third-party software.
