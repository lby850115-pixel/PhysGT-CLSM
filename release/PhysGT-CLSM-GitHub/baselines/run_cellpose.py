"""run_cellpose_server.py

Cellpose cyto2 inference on MITO DATA_1 images.

Usage:
    python run_cellpose_server.py
    python run_cellpose_server.py --diameter 15
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

ROOT    = Path(__file__).resolve().parent
IMG_DIR = ROOT / "MITO DATA_1"
OUT_DIR = ROOT / "predictions" / "cellpose"
FIG_DIR = ROOT / "figures" / "cellpose"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MITO_CHANNEL = 1


def load_channel(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, MITO_CHANNEL].astype(np.float32)
    if img.ndim == 3 and img.shape[0] <= 4:
        return img[MITO_CHANNEL].astype(np.float32)
    return img.astype(np.float32)


def save_figure(raw: np.ndarray, instances: np.ndarray,
                stem: str, n_mito: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.suptitle(f"{stem}  |  Cellpose  |  n={n_mito} mitochondria",
                 fontsize=10)
    vmax = np.percentile(raw[raw > 0], 99.5) if raw.max() > 0 else 1.0
    raw_norm = np.clip(raw / vmax, 0, 1)

    axes[0].imshow(raw_norm, cmap='gray')
    axes[0].set_title('Raw (mito channel)')
    axes[0].axis('off')

    axes[1].imshow(instances, cmap='nipy_spectral')
    axes[1].set_title(f'Instances  n={n_mito}')
    axes[1].axis('off')

    axes[2].imshow(raw_norm, cmap='gray')
    bd = find_boundaries(instances, mode='outer')
    ov = np.zeros((*raw.shape, 4), dtype=np.float32)
    ov[bd] = [0.2, 0.9, 0.2, 0.9]
    axes[2].imshow(ov)
    axes[2].set_title('Boundary overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / f"{stem}_cellpose.png"), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="bact_fluor_cp3",
                        help="Cellpose model (bact_fluor_cp3 / cyto2 / cyto3)")
    parser.add_argument("--diameter",  type=float, default=0,
                        help="Object diameter in px (0=auto-estimate)")
    parser.add_argument("--flow_thr",  type=float, default=0.4)
    parser.add_argument("--prob_thr",  type=float, default=0.0)
    parser.add_argument("--min_area",  type=int,   default=5)
    args = parser.parse_args()

    import torch
    from cellpose import models

    print("=" * 60)
    print(f"Cellpose  model={args.model}  (Stringer et al., Nature Methods 2021)")
    print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    use_gpu = torch.cuda.is_available()
    # CellposeModel supports all model names including _cp3 variants
    # Cellpose wrapper only works for classic models (cyto/cyto2/nuclei)
    model = models.CellposeModel(model_type=args.model, gpu=use_gpu)

    tifs = sorted(IMG_DIR.glob("*.tif"))
    if not tifs:
        print(f"[ERROR] No TIF files in {IMG_DIR}")
        sys.exit(1)

    print(f"\nProcessing {len(tifs)} images...")
    print(f"  model={args.model}  diameter={'auto' if args.diameter == 0 else args.diameter}  "
          f"flow_thr={args.flow_thr}  prob_thr={args.prob_thr}")
    print(f"{'#':<3} {'File':<55} {'N':>5} {'diam_est':>9} {'area_med':>9}")
    print("-" * 85)

    stats = []
    for i, fpath in enumerate(tifs, 1):
        raw = load_channel(fpath)

        # Cellpose expects float32 [0,1] or uint8
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.8)
        img_norm = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

        # channels=[0,0]: grayscale input, no separate nucleus channel
        # CellposeModel.eval returns (masks, flows, styles) — no diams
        masks, flows, styles = model.eval(
            img_norm,
            diameter=args.diameter if args.diameter > 0 else None,
            channels=[0, 0],
            flow_threshold=args.flow_thr,
            cellprob_threshold=args.prob_thr,
        )
        diam_est = round(float(args.diameter) if args.diameter > 0 else 0.0, 1)

        # Remove small objects
        instances = masks.astype(np.uint16)
        for prop in regionprops(instances):
            if prop.area < args.min_area:
                instances[instances == prop.label] = 0
        instances, n = nd_label(instances > 0)
        instances = instances.astype(np.uint16)

        areas = [p.area for p in regionprops(instances)]
        area_med = int(np.median(areas)) if areas else 0

        tifffile.imwrite(str(OUT_DIR / fpath.name), instances)
        save_figure(raw, instances, fpath.stem, n)

        stats.append({
            "file":       fpath.name,
            "n_mito":     n,
            "diam_est":   diam_est,
            "area_median": area_med,
            "area_mean":  round(float(np.mean(areas)), 1) if areas else 0,
        })
        print(f"{i:<3} {fpath.name:<55} {n:>5} {diam_est:>9} {area_med:>9}")

    csv_path = ROOT / "results" / "cellpose_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","n_mito","diam_est",
                                           "area_median","area_mean"])
        w.writeheader()
        w.writerows(stats)

    print(f"\nInstance TIFs -> {OUT_DIR}")
    print(f"Overlay PNGs  -> {FIG_DIR}")
    print(f"Stats CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
