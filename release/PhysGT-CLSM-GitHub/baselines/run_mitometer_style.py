"""run_mitometer.py

Re-implementation of the Mitometer segmentation algorithm
(Lam et al., Nature Methods 2021, doi:10.1038/s41592-021-01234-z).

Algorithm (no deep learning, no GT required):
  1. Diffuse background subtraction via multi-scale circular median filters
  2. Parameter exploration: auto-select Gaussian sigma + intensity threshold
     by minimising inter-frame variability in mito count/area
  3. Gaussian smooth → threshold → connected-component instance segmentation

For static single images (no time-lapse), step 2 is replaced by a
grid search that maximises detected objects while minimising noise.

Usage
-----
    python run_mitometer.py
    python run_mitometer.py --channel 1 --min_area 5 --max_area 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import (gaussian_filter, label as nd_label,
                           median_filter)
from skimage.morphology import disk
from skimage.filters import rank
from skimage.measure import regionprops

ROOT     = Path(__file__).resolve().parent
IMG_DIR  = ROOT / "MITO DATA_1"
OUT_DIR  = ROOT / "predictions" / "mitometer"
FIG_DIR  = ROOT / "figures" / "mitometer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MITO_CHANNEL = 1   # green channel carries mito signal


# ── Step 1: Diffuse background subtraction ──────────────────────────────────

def diffuse_background_subtract(img: np.ndarray,
                                 r_min: int = 3,
                                 r_max: int = 10) -> np.ndarray:
    """
    Mitometer Fig.1b / Extended Data Fig.1:
    Convolve circular median filters of increasing radii (r_min..r_max px),
    take pixel-wise minimum across all filtered images (including original),
    subtract that minimum image from the original.
    """
    stack = [img.astype(np.float32)]
    for r in range(r_min, r_max + 1):
        filtered = rank.median(
            np.clip(img, 0, 255).astype(np.uint8), disk(r)
        ).astype(np.float32)
        stack.append(filtered)
    bg = np.min(np.stack(stack, axis=0), axis=0)
    return np.clip(img.astype(np.float32) - bg, 0, None)


# ── Step 2: Parameter exploration ───────────────────────────────────────────

def _segment_once(img_db: np.ndarray, sigma: float,
                  thresh: float, min_area: int, max_area: int) -> np.ndarray:
    """Apply Gaussian + threshold + size filter, return binary mask."""
    smoothed = gaussian_filter(img_db, sigma=sigma)
    binary   = smoothed > thresh
    labeled, _ = nd_label(binary)
    # remove objects outside area bounds
    for prop in regionprops(labeled):
        if prop.area < min_area or prop.area > max_area:
            labeled[labeled == prop.label] = 0
    return (labeled > 0).astype(np.uint8)


def explore_parameters(img_db: np.ndarray,
                        min_area: int, max_area: int,
                        sigmas: list[float] | None = None,
                        n_thresh: int = 20) -> tuple[float, float]:
    """
    Mitometer Fig.1c / Extended Data Fig.2:
    For static images (no time-lapse), score each (sigma, thresh) pair by:
      score = n_objects - variability_penalty
    where variability_penalty penalises fragmented or merged objects.
    Returns (best_sigma, best_thresh).
    """
    if sigmas is None:
        sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    vmax = np.percentile(img_db[img_db > 0], 99.0) if img_db.max() > 0 else 1.0
    thresholds = np.linspace(vmax * 0.02, vmax * 0.5, n_thresh)

    best_score  = -np.inf
    best_sigma  = sigmas[1]
    best_thresh = thresholds[n_thresh // 3]

    for sigma in sigmas:
        smoothed = gaussian_filter(img_db, sigma=sigma)
        for thresh in thresholds:
            binary = smoothed > thresh
            labeled, n = nd_label(binary)
            if n == 0:
                continue
            areas = np.array([p.area for p in regionprops(labeled)])
            valid = areas[(areas >= min_area) & (areas <= max_area)]
            if len(valid) == 0:
                continue
            # score: many valid objects, low area CoV (stable segmentation)
            cov   = valid.std() / valid.mean() if valid.mean() > 0 else 999
            score = len(valid) - cov * 2
            if score > best_score:
                best_score  = score
                best_sigma  = sigma
                best_thresh = thresh

    return best_sigma, best_thresh


# ── Step 3: Instance segmentation ───────────────────────────────────────────

def segment_image(img: np.ndarray,
                  min_area: int = 5,
                  max_area: int = 2000) -> tuple[np.ndarray, dict]:
    """Full Mitometer pipeline for a single 2-D grayscale image."""
    img_db = diffuse_background_subtract(img)
    sigma, thresh = explore_parameters(img_db, min_area, max_area)

    smoothed = gaussian_filter(img_db, sigma=sigma)
    binary   = smoothed > thresh
    labeled, _ = nd_label(binary)

    # size filter
    for prop in regionprops(labeled):
        if prop.area < min_area or prop.area > max_area:
            labeled[labeled == prop.label] = 0

    # re-label consecutively
    instances, n = nd_label(labeled > 0)
    instances = instances.astype(np.uint16)

    props = regionprops(instances)
    areas       = [p.area        for p in props]
    major_axes  = [p.major_axis_length for p in props]
    minor_axes  = [p.minor_axis_length for p in props]
    solidities  = [p.solidity    for p in props]

    stats = {
        "n_mito":        n,
        "sigma":         round(sigma, 2),
        "thresh":        round(float(thresh), 3),
        "area_mean":     round(float(np.mean(areas)),  1) if areas else 0,
        "area_median":   round(float(np.median(areas)),1) if areas else 0,
        "area_std":      round(float(np.std(areas)),   1) if areas else 0,
        "major_ax_mean": round(float(np.mean(major_axes)), 1) if major_axes else 0,
        "minor_ax_mean": round(float(np.mean(minor_axes)), 1) if minor_axes else 0,
        "aspect_ratio":  round(float(np.mean(
            [mn/mj for mn, mj in zip(minor_axes, major_axes) if mj > 0]
        )), 3) if major_axes else 0,
        "solidity_mean": round(float(np.mean(solidities)), 3) if solidities else 0,
    }
    return instances, stats


# ── Visualisation ────────────────────────────────────────────────────────────

def save_figure(raw: np.ndarray, instances: np.ndarray,
                stats: dict, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=130)
    fig.suptitle(
        f"{stem}  |  n={stats['n_mito']} mito  |  "
        f"area_med={stats['area_median']} px  |  "
        f"aspect={stats['aspect_ratio']}  (minor/major)",
        fontsize=10
    )

    vmax = np.percentile(raw[raw > 0], 99.5) if raw.max() > 0 else 1.0
    raw_norm = np.clip(raw / vmax, 0, 1)

    axes[0].imshow(raw_norm, cmap='gray')
    axes[0].set_title('Raw (mito channel)')
    axes[0].axis('off')

    axes[1].imshow(instances, cmap='nipy_spectral')
    axes[1].set_title(f'Instance segmentation  n={stats["n_mito"]}')
    axes[1].axis('off')

    # overlay: boundaries on raw
    from skimage.segmentation import find_boundaries
    axes[2].imshow(raw_norm, cmap='gray')
    bd = find_boundaries(instances, mode='outer')
    ov = np.zeros((*raw.shape, 4), dtype=np.float32)
    ov[bd] = [0.0, 1.0, 0.4, 0.9]
    axes[2].imshow(ov)
    axes[2].set_title('Boundary overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / f"{stem}_mitometer.png"), bbox_inches='tight')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel",  type=int,   default=MITO_CHANNEL)
    parser.add_argument("--min_area", type=int,   default=5)
    parser.add_argument("--max_area", type=int,   default=2000)
    args = parser.parse_args()

    tifs = sorted(IMG_DIR.glob("*.tif"))
    if not tifs:
        print(f"[ERROR] No TIF files in {IMG_DIR}")
        sys.exit(1)

    print(f"Mitometer segmentation  |  channel={args.channel}  "
          f"min_area={args.min_area}  max_area={args.max_area}")
    print(f"{'#':<3} {'File':<55} {'N':>5} {'σ':>5} {'area_med':>9} {'aspect':>8}")
    print("-" * 90)

    all_stats = []
    for i, fpath in enumerate(tifs, 1):
        img_rgb = tifffile.imread(str(fpath))
        # extract mito channel
        if img_rgb.ndim == 3 and img_rgb.shape[2] >= 3:
            raw = img_rgb[:, :, args.channel].astype(np.float32)
        elif img_rgb.ndim == 3 and img_rgb.shape[0] <= 4:
            raw = img_rgb[args.channel].astype(np.float32)
        else:
            raw = img_rgb.astype(np.float32)

        instances, stats = segment_image(raw, args.min_area, args.max_area)

        tifffile.imwrite(str(OUT_DIR / fpath.name), instances)
        save_figure(raw, instances, stats, fpath.stem)

        stats["file"] = fpath.name
        all_stats.append(stats)
        print(f"{i:<3} {fpath.name:<55} {stats['n_mito']:>5} "
              f"{stats['sigma']:>5} {stats['area_median']:>9} "
              f"{stats['aspect_ratio']:>8}")

    # summary CSV
    import csv
    csv_path = ROOT / "results" / "mitometer_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "n_mito", "sigma", "thresh", "area_mean", "area_median",
              "area_std", "major_ax_mean", "minor_ax_mean", "aspect_ratio",
              "solidity_mean"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_stats)

    print(f"\nInstance TIFs  -> {OUT_DIR}")
    print(f"Overlay PNGs   -> {FIG_DIR}")
    print(f"Stats CSV      -> {csv_path}")


if __name__ == "__main__":
    main()
