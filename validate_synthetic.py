"""
validate_synthetic.py
=====================
Synthetic self-validation for PhysGT-CLSM.

Generates N_TILES synthetic CLSM-like tiles using the physics pipeline,
segments each with the PhysGT algorithm, and computes Dice / AJI / F1@0.5
against the known physics GT.

Physics functions are inlined from PhysGT_CLSM.py to avoid the module-level
input() call that runs when that file is imported.

Output:
  results/synthetic_validation.csv          per-tile metrics
  results/synthetic_validation_summary.txt  mean ± SD validation summary
"""

import math, csv, warnings
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')

ROOT    = Path(__file__).resolve().parent
OUT_RES = ROOT / 'results'
OUT_RES.mkdir(parents=True, exist_ok=True)

N_TILES = 100

# ── Physics constants (from PhysGT_CLSM.py) ──────────────────────────────────
PIXEL_NM      = 120.25
NA            = 1.2
WAVELENGTH    = 488.0
PSF_SIGMA_NM  = 0.61 * WAVELENGTH / NA / 2.355
PSF_SIGMA_PX  = PSF_SIGMA_NM / PIXEL_NM
READOUT_SIGMA = 8.0
POISSON_SCALE = 80.0
MITO_DIAM_NM       = 250.0
MITO_DIAM_PX       = MITO_DIAM_NM / PIXEL_NM
# In CLSM we observe the lateral extent (length), not the diameter.
# Shortest observable mito ~1 µm; typical rod 1–3 µm = 8–25 px.
MITO_LENGTH_MIN_NM = 1000.0
MITO_LENGTH_MIN_PX = MITO_LENGTH_MIN_NM / PIXEL_NM   # ≈ 8.3 px
TILE          = 128
OUTPUT_SZ     = 256
P_DOT         = 0.25
P_ROD         = 0.45


# ── Physics pipeline (inlined from PhysGT_CLSM.py) ───────────────────────────

def _draw_thick_line(canvas, r0, c0, r1, c1, radius_px):
    steps = max(int(math.hypot(r1-r0, c1-c0)) * 2 + 1, 2)
    for t in np.linspace(0, 1, steps):
        r = r0 + t * (r1 - r0)
        c = c0 + t * (c1 - c0)
        rr = int(round(r)); cc = int(round(c))
        rad = int(math.ceil(radius_px))
        for dr in range(-rad, rad+1):
            for dc in range(-rad, rad+1):
                if dr*dr + dc*dc <= radius_px**2:
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]:
                        canvas[nr, nc] = 1.0


def make_mito_geometry(shape, mtype, rng):
    H, W = shape
    canvas = np.zeros((H, W), dtype=np.float32)
    r_px = max(1.0, MITO_DIAM_PX / 2)
    margin = int(r_px) + 4

    if mtype == 'dot':
        cr = rng.integers(margin, H-margin)
        cc = rng.integers(margin, W-margin)
        for dr in range(-int(r_px)-2, int(r_px)+3):
            for dc in range(-int(r_px)-2, int(r_px)+3):
                if dr*dr + dc*dc <= r_px**2:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        canvas[nr, nc] = 1.0
    elif mtype == 'rod':
        length_px = rng.uniform(3*MITO_DIAM_PX, 12*MITO_DIAM_PX)
        angle = rng.uniform(0, math.pi)
        cr = rng.integers(margin+int(length_px//2), H-margin-int(length_px//2))
        cc = rng.integers(margin+int(length_px//2), W-margin-int(length_px//2))
        dr = math.sin(angle) * length_px / 2
        dc = math.cos(angle) * length_px / 2
        _draw_thick_line(canvas, cr-dr, cc-dc, cr+dr, cc+dc, r_px)
    else:  # network
        n_branches = rng.integers(2, 5)
        cr = rng.integers(margin*2, H-margin*2)
        cc = rng.integers(margin*2, W-margin*2)
        angles = rng.uniform(0, 2*math.pi, n_branches)
        for ang in angles:
            length_px = rng.uniform(4*MITO_DIAM_PX, 10*MITO_DIAM_PX)
            er = np.clip(cr + math.sin(ang) * length_px, margin, H-margin)
            ec = np.clip(cc + math.cos(ang) * length_px, margin, W-margin)
            _draw_thick_line(canvas, cr, cc, er, ec, r_px)

    return canvas, canvas > 0


def place_fluorophores(emitter_mask, rng, density=0.6):
    flip = rng.random(emitter_mask.shape).astype(np.float32)
    return (emitter_mask * (flip < density)).astype(np.float32)


def simulate_image(fluorophore_map, rng):
    blurred = gaussian_filter(fluorophore_map * POISSON_SCALE, sigma=PSF_SIGMA_PX)
    noisy = rng.poisson(np.clip(blurred, 0, None)).astype(np.float32)
    noisy += rng.normal(0, READOUT_SIGMA, noisy.shape).astype(np.float32)
    return noisy


# ── Tile generator with per-instance GT ──────────────────────────────────────

def make_tile_with_instances(rng, n_mito_range=(1, 5)):
    """
    Returns (image_256, gt_binary_256, gt_instance_256).
    gt_instance_256: uint16, each mitochondrion has a unique integer label.
    """
    big_img  = np.zeros((OUTPUT_SZ, OUTPUT_SZ), dtype=np.float32)
    big_bin  = np.zeros((OUTPUT_SZ, OUTPUT_SZ), dtype=np.uint8)
    big_inst = np.zeros((OUTPUT_SZ, OUTPUT_SZ), dtype=np.uint16)
    inst_id  = 1

    for ti in range(2):
        for tj in range(2):
            sub_img  = np.zeros((TILE, TILE), dtype=np.float32)
            sub_bin  = np.zeros((TILE, TILE), dtype=np.uint8)
            sub_inst = np.zeros((TILE, TILE), dtype=np.uint16)
            n_mito   = rng.integers(*n_mito_range)

            for _ in range(n_mito):
                p = rng.random()
                if p < P_DOT:
                    mtype = 'dot'
                elif p < P_DOT + P_ROD:
                    mtype = 'rod'
                else:
                    mtype = 'network'

                emitter_mask, binary_mask = make_mito_geometry((TILE, TILE), mtype, rng)
                fluor_map = place_fluorophores(emitter_mask, rng)
                sub_img += simulate_image(fluor_map, rng)
                # Dilate GT by 1 px: raw emitter mask is ~2 px wide, but the
                # PSF-convolved image (FWHM≈2 px) makes the observable extent ~4 px.
                # Evaluating against the raw mask puts IoU right at the 0.5 boundary.
                gt_mask = binary_dilation(binary_mask, disk(1))
                sub_bin  = np.maximum(sub_bin, gt_mask.astype(np.uint8))
                sub_inst[gt_mask] = inst_id
                inst_id += 1

            r0, c0 = ti*TILE, tj*TILE
            big_img [r0:r0+TILE, c0:c0+TILE] = sub_img
            big_bin [r0:r0+TILE, c0:c0+TILE] = sub_bin
            big_inst[r0:r0+TILE, c0:c0+TILE] = sub_inst

    return big_img, big_bin, big_inst


# ── Segmentation on numpy array ───────────────────────────────────────────────

def segment_array(image):
    """Runs the PhysGT segmentation pipeline on a float32 numpy array."""
    lo, hi = np.percentile(image, [1, 99])
    norm = np.clip((image - lo) / (hi - lo + 1e-9), 0, 1)

    # sigma=1.0 suppresses sub-pixel noise without broadening 2px-wide structures
    smoothed = gaussian_filter(norm, sigma=1.0)

    thresh = threshold_triangle(smoothed)
    binary = binary_closing(smoothed > thresh, disk(1))

    dist = distance_transform_edt(binary)
    # A 2px-wide rod has a flat ridge in the distance transform (max ≈1 px).
    # Smoothing collapses the ridge into one broad peak so a single rod gets
    # one watershed marker instead of being split every ~min_dist pixels.
    dist_smooth = gaussian_filter(dist, sigma=3.0)
    min_dist = max(10, int(MITO_LENGTH_MIN_PX * 1.2))
    coords   = peak_local_max(dist_smooth, min_distance=min_dist, labels=binary)
    markers  = np.zeros(binary.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    labeled_ws = watershed(-dist, markers, mask=binary)

    # Minimum area = half the footprint of the shortest observable rod
    MIN_AREA = max(20, int(MITO_LENGTH_MIN_PX * MITO_DIAM_PX * 0.5))
    labeled  = np.zeros_like(labeled_ws, dtype=np.uint16)
    new_id   = 1
    for iid in range(1, labeled_ws.max() + 1):
        mask = labeled_ws == iid
        if mask.sum() >= MIN_AREA:
            labeled[mask] = new_id
            new_id += 1

    return labeled


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_semantic(pred_labels, gt_binary):
    pred_bin = (pred_labels > 0).astype(np.uint8)
    tp = int(np.logical_and(pred_bin, gt_binary).sum())
    fp = int(np.logical_and(pred_bin, ~gt_binary.astype(bool)).sum())
    fn = int(np.logical_and(~pred_bin.astype(bool), gt_binary).sum())
    denom = 2*tp + fp + fn
    return (2*tp / denom) if denom > 0 else 1.0


def _iou_matrix(pred_labels, gt_labels):
    """Returns IoU for every (gt_id, pred_id) pair as a dict."""
    gt_ids   = [i for i in np.unique(gt_labels)   if i > 0]
    pred_ids = [i for i in np.unique(pred_labels) if i > 0]
    iou = {}
    for g in gt_ids:
        gm = gt_labels == g
        for p in pred_ids:
            pm = pred_labels == p
            inter = int(np.logical_and(gm, pm).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(gm, pm).sum())
            iou[(g, p)] = inter / union
    return iou, gt_ids, pred_ids


def aji(pred_labels, gt_labels):
    """Aggregated Jaccard Index (Kumar et al. 2017)."""
    iou_dict, gt_ids, pred_ids = _iou_matrix(pred_labels, gt_labels)
    if not gt_ids:
        return 1.0 if not pred_ids else 0.0

    matched_pred = set()
    num = 0; denom = 0
    for g in gt_ids:
        gm = gt_labels == g
        best_p, best_iou = None, 0.0
        for p in pred_ids:
            v = iou_dict.get((g, p), 0.0)
            if v > best_iou:
                best_iou = v; best_p = p
        if best_p is not None:
            pm = pred_labels == best_p
            num   += int(np.logical_and(gm, pm).sum())
            denom += int(np.logical_or(gm, pm).sum())
            matched_pred.add(best_p)
        else:
            denom += int(gm.sum())

    for p in pred_ids:
        if p not in matched_pred:
            denom += int((pred_labels == p).sum())

    return num / denom if denom > 0 else 0.0


def f1_at_iou(pred_labels, gt_labels, iou_thresh=0.5):
    iou_dict, gt_ids, pred_ids = _iou_matrix(pred_labels, gt_labels)
    if not gt_ids and not pred_ids:
        return 1.0

    matched_gt = set(); matched_pred = set()
    for g in gt_ids:
        for p in pred_ids:
            if iou_dict.get((g, p), 0.0) >= iou_thresh:
                if g not in matched_gt and p not in matched_pred:
                    matched_gt.add(g); matched_pred.add(p)

    tp = len(matched_gt)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids)  - len(matched_gt)
    denom = 2*tp + fp + fn
    return (2*tp / denom) if denom > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print(f'PhysGT-CLSM Synthetic Self-Validation  (N={N_TILES} tiles)')
    print(f'PSF sigma = {PSF_SIGMA_PX:.3f} px  |  pixel = {PIXEL_NM} nm')
    print('=' * 60)

    rng = np.random.default_rng(42)
    rows = []

    for i in range(N_TILES):
        img, gt_bin, gt_inst = make_tile_with_instances(rng)
        pred = segment_array(img)

        n_gt   = int(gt_inst.max())
        n_pred = int(pred.max())
        d  = dice_semantic(pred, gt_bin)
        aj = aji(pred, gt_inst)
        f1 = f1_at_iou(pred, gt_inst, 0.5)

        rows.append({'tile': i, 'n_gt': n_gt, 'n_pred': n_pred,
                     'dice': d, 'aji': aj, 'f1_05': f1})

        if (i + 1) % 10 == 0:
            print(f'  [{i+1:3d}/{N_TILES}]  '
                  f'Dice={d:.3f}  AJI={aj:.3f}  F1@0.5={f1:.3f}  '
                  f'GT={n_gt}  Pred={n_pred}')

    # Save per-tile CSV
    csv_path = OUT_RES / 'synthetic_validation.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Compute summary
    def _ms(key):
        vals = [r[key] for r in rows]
        return np.mean(vals), np.std(vals)

    dice_m,  dice_s  = _ms('dice')
    aji_m,   aji_s   = _ms('aji')
    f1_m,    f1_s    = _ms('f1_05')
    ngt_m,   ngt_s   = _ms('n_gt')
    npred_m, npred_s = _ms('n_pred')

    summary = (
        f"\n{'=' * 60}\n"
        f"SYNTHETIC SELF-VALIDATION SUMMARY  (N={N_TILES} tiles, seed=42)\n"
        f"{'=' * 60}\n"
        f"  Dice (semantic)  : {dice_m:.3f} +/- {dice_s:.3f}\n"
        f"  AJI              : {aji_m:.3f} +/- {aji_s:.3f}\n"
        f"  F1 @ IoU=0.5     : {f1_m:.3f} +/- {f1_s:.3f}\n"
        f"  GT count / tile  : {ngt_m:.1f} +/- {ngt_s:.1f}\n"
        f"  Pred count / tile: {npred_m:.1f} +/- {npred_s:.1f}\n"
        f"{'=' * 60}\n"
        f"Per-tile CSV: {csv_path}\n"
    )
    print(summary)

    txt_path = OUT_RES / 'synthetic_validation_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'Summary saved: {txt_path}')
