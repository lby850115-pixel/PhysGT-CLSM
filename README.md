# PhysGT-CLSM

**Physics-Informed Annotation-Free Ground Truth Generation for CLSM Mitochondria Segmentation**

> Submitted to *Biomedical Signal Processing and Control* (BSPC)

PhysGT-CLSM is a fully annotation-free, GPU-free pipeline for mitochondria instance segmentation in confocal laser scanning microscopy (CLSM) images. It adapts the physics-based simulation framework of [Sekh et al. (2021)](https://doi.org/10.1038/s42256-021-00420-0) with three CLSM-specific contributions:

1. **PSF-adaptive pixel calibration** — lateral pixel size derived from optical parameters (λ, NA)
2. **Distance-transform watershed** — instance separation without learned weights
3. **Annotation-free direct inference** — no U-Net, no MATLAB, no GPU required

---

## Requirements

- Python 3.11
- See `requirements.txt` for package versions

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n physgt python=3.11
conda activate physgt
pip install -r requirements.txt
```

---

## Dataset

The study uses 34 CLSM mitochondria images across three human cancer cell lines:

| Cell line | N images | Source |
|-----------|----------|--------|
| HeLa      | 20       | `MITO DATA/HELA/` |
| BxPC-3    | 6        | `MITO DATA/BXPC-3/` |
| MCF-7     | 8        | `MITO DATA/MCF-7/` |

**Acquisition parameters:** 1024 × 1024 px, pixel size = 120.25 nm/px, NA = 1.2, λ_ex = 488 nm.

Raw images are available at: *(DOI to be assigned upon acceptance)*

---

## Repository Structure

```
PhysGT-CLSM/
├── PhysGT_CLSM.py              # Core model: physics simulation + segmentation
├── rerun_physegt_34.py         # Apply segmentation to all 34 images
├── validate_synthetic.py       # Synthetic self-validation (Dice / AJI / F1)
├── stats_significance.py       # Kruskal-Wallis + Dunn post-hoc tests
├── 08_morphology_34.py         # Morphology quantification (6 metrics × 6 models)
├── paper_figures.py            # Reproduce all paper figures (Fig. 1–4, S1–S2)
├── gen_flowchart.py            # Pipeline flowchart (SVG/PDF)
├── macros/
│   ├── fill_rois_as_labels.ijm # Fiji macro: ROI → label map
│   └── load_pred_rois.ijm      # Fiji macro: load predictions as ROIs
├── results/                    # CSV outputs (auto-generated)
├── figures/                    # Figure outputs (auto-generated)
│   └── paper/                  # Publication-ready figures
├── predictions_34/             # Instance label TIFs (auto-generated)
├── requirements.txt
└── LICENSE
```

---

## Usage

### 1. Run PhysGT-CLSM on your own images

```bash
python PhysGT_CLSM.py --img_dir /path/to/your/tifs
```

Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_dir` | `MITO DATA_1/` | Input directory of `.tif` files |
| `--out_pred` | `predictions/physegt_clsm/` | Output label map directory |
| `--out_fig`  | `figures/physegt_clsm/` | Output QC figure directory |
| `--out_res`  | `results/` | Output CSV directory |
| `--min_area` | auto (86 px²) | Override minimum instance area |

Output per image:
- `predictions/physegt_clsm/<name>.tif` — uint16 instance label map
- `figures/physegt_clsm/<name>_overlay.png` — 3-panel QC figure (raw / binary / instances)
- `results/physegt_clsm_stats.csv` — per-image count and area statistics

### 2. Reproduce paper results (34-image dataset)

```bash
# Step 1: Segment all 34 images
python rerun_physegt_34.py

# Step 2: Morphology quantification (all 6 models)
python 08_morphology_34.py

# Step 3: Statistical tests
python stats_significance.py

# Step 4: Generate all paper figures
python paper_figures.py
```

### 3. Synthetic self-validation

```bash
python validate_synthetic.py
```

Generates 100 synthetic tile pairs (image + physics GT) and evaluates:
- Dice coefficient
- Aggregated Jaccard Index (AJI)
- F1 score at IoU threshold 0.5

---

## Physics Parameters

All segmentation parameters are derived from optical first principles — no tuning on the test set.

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Pixel size | 120.25 nm/px | Microscope calibration |
| PSF σ | 0.876 px | σ = 0.61 λ / NA / 2.355 = 105.3 nm |
| Pre-smooth σ | 1.0 px | ≈ 1 PSF width (readout noise suppression) |
| min_distance | 10 px | 1.2 × L_min = 1.2 × 8.3 px (L_min = 1 µm) |
| MIN_AREA | 20 px² | ½ × L_min × d_mito = ½ × 8.3 × 2.08 px² |

---

## Results

Synthetic self-validation (N = 100 tiles):

| Metric | Mean ± SD |
|--------|-----------|
| Dice | 0.847 ± 0.041 |
| AJI | 0.650 ± 0.106 |
| F1 @ IoU=0.5 | 0.740 ± 0.149 |

Cross-model comparison (34 images, Kruskal-Wallis): all 6 morphology metrics show significant inter-model differences (H = 145–187, p < 10⁻²⁹). PhysGT-CLSM is not significantly different from MitoSegNet on Area, Aspect Ratio, Eccentricity, and Solidity (Dunn BH-corrected, p > 0.05).

---

## Citation

If you use this code, please cite:

```bibtex
@article{physgt_clsm_2025,
  title   = {Physics-Informed Annotation-Free Ground Truth Generation for
             CLSM Mitochondria Segmentation: A Lateral Validation Against
             Five Mainstream Models},
  journal = {Biomedical Signal Processing and Control},
  year    = {2025},
  note    = {Under review}
}
```

This work builds on:

```bibtex
@article{sekh2021physics,
  title   = {Physics-based machine learning for subcellular segmentation in living cells},
  author  = {Sekh, Arif Ahmed and others},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  pages   = {1071--1080},
  year    = {2021},
  doi     = {10.1038/s42256-021-00420-0}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
