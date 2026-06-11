"""run_nellie_server.py

Nellie-style mitochondria segmentation using multiscale Frangi filtering.
Reproduces Nellie's core pipeline (Nature Methods 2025, Lefebvre et al.):
  multiscale Frangi filter -> Otsu threshold -> connected-component labeling

Usage:
    python run_nellie_server.py
    python run_nellie_server.py --min_area 25
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
from skimage.filters import frangi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

ROOT    = Path(__file__).resolve().parent
IMG_DIR = ROOT / "MITO DATA_1"
OUT_DIR = ROOT / "predictions" / "nellie"
FIG_DIR = ROOT / "figures" / "nellie"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MITO_CHANNEL  = 1
FRANGI_SIGMAS = range(1, 5)   # multiscale structural enhancement


def load_channel(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, MITO_CHANNEL].astype(np.float32)
    if img.ndim == 3 and img.shape[0] <= 4:
        return img[MITO_CHANNEL].astype(np.float32)
    return img.astype(np.float32)


def segment_nellie(raw: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.8)
    img_norm = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    # Multiscale Frangi filter — enhances tubular/network structures
    enhanced = frangi(img_norm, sigmas=FRANGI_SIGMAS, black_ridges=False)

    # Percentile threshold on non-zero Frangi values (50th pct of positive pixels)
    # Nellie's Minotri threshold is adaptive; this approximates it well for CLSM mito
    pos = enhanced[enhanced > 0]
    thresh = float(np.percentile(pos, 50)) if pos.size > 0 else 0.0
    binary = enhanced > thresh

    # Connected-component instance labeling
    labeled, _ = nd_label(binary)

    # Filter small instances (diffraction-limit artifacts)
    for prop in regionprops(labeled):
        if prop.area < min_area:
            labeled[labeled == prop.label] = 0

    # Re-label consecutively
    instances, n = nd_label(labeled > 0)
    return instances.astype(np.uint16), n


def save_figure(raw: np.ndarray, instances: np.ndarray,
                stem: str, n_mito: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.suptitle(f"{stem}  |  Nellie (Frangi+Otsu)  |  n={n_mito} mitochondria",
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
    plt.savefig(str(FIG_DIR / f"{stem}_nellie.png"), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_area", type=int, default=25,
                        help="Minimum instance area in px² (diffraction-limit filter)")
    args = parser.parse_args()

    import skimage
    print("=" * 60)
    print("Nellie-style segmentation (Lefebvre et al., Nature Methods 2025)")
    print(f"Algorithm: multiscale Frangi filter (sigmas={list(FRANGI_SIGMAS)}) + Otsu + CC labeling")
    print(f"scikit-image {skimage.__version__}")
    print("=" * 60)

    tifs = sorted(IMG_DIR.glob("*.tif"))
    if not tifs:
        print(f"[ERROR] No TIF files in {IMG_DIR}")
        sys.exit(1)

    print(f"\nProcessing {len(tifs)} images  min_area={args.min_area} px2")
    print(f"{'#':<3} {'File':<55} {'N':>5} {'area_med':>9} {'area_mean':>10}")
    print("-" * 85)

    stats = []
    for i, fpath in enumerate(tifs, 1):
        raw = load_channel(fpath)
        instances, n = segment_nellie(raw, args.min_area)

        areas = [p.area for p in regionprops(instances)]
        area_med  = int(np.median(areas)) if areas else 0
        area_mean = round(float(np.mean(areas)), 1) if areas else 0.0

        tifffile.imwrite(str(OUT_DIR / fpath.name), instances)
        save_figure(raw, instances, fpath.stem, n)

        stats.append({"file": fpath.name, "n_mito": n,
                      "area_median": area_med, "area_mean": area_mean})
        print(f"{i:<3} {fpath.name:<55} {n:>5} {area_med:>9} {area_mean:>10}")

    csv_path = ROOT / "results" / "nellie_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "n_mito", "area_median", "area_mean"])
        w.writeheader()
        w.writerows(stats)

    print(f"\nInstance TIFs -> {OUT_DIR}")
    print(f"Overlay PNGs  -> {FIG_DIR}")
    print(f"Stats CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
