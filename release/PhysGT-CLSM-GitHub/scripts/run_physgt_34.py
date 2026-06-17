"""Re-run PhysGT-CLSM on the full manuscript image set.

The filename keeps the earlier internal `_34` suffix, but the final manuscript
analysis uses the curated 33-image set. The parameters below match the final
real-image working preset used by `src/physgt_clsm.py`.
"""
import math, warnings
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.filters import threshold_triangle
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / 'predictions_33' / 'physegt_clsm'
PRED_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIRS = {
    'HELA':  ROOT / 'MITO DATA' / 'HELA',
    'BXPC3': ROOT / 'MITO DATA' / 'BXPC-3',
    'MCF7':  ROOT / 'MITO DATA' / 'MCF-7',
}

PIXEL_NM           = 120.25
MITO_DIAM_PX       = 250.0  / PIXEL_NM   # 2.08 px
MITO_LENGTH_MIN_PX = 1000.0 / PIXEL_NM   # 8.3 px


def segment(fp):
    raw = tifffile.imread(str(fp))
    if raw.ndim == 2:
        mito = raw.astype(np.float32)
    elif raw.ndim == 3:
        # auto-detect the channel with the strongest signal
        ch_means = [raw[:, :, i].mean() for i in range(raw.shape[2])]
        best_ch = int(np.argmax(ch_means))
        mito = raw[:, :, best_ch].astype(np.float32)
    else:
        mito = raw[..., 1].astype(np.float32)

    lo, hi = np.percentile(mito, [1, 99])
    norm = np.clip((mito - lo) / (hi - lo + 1e-9), 0, 1)
    smoothed = gaussian_filter(norm, sigma=1.0)
    thresh = threshold_triangle(smoothed) * 1.35
    binary = smoothed > thresh

    dist = distance_transform_edt(binary)
    dist_smooth = gaussian_filter(dist, sigma=1.2)
    min_dist = 8
    coords = peak_local_max(dist_smooth, min_distance=min_dist, labels=binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    labeled_ws = watershed(-dist, markers, mask=binary)
    MIN_AREA = 20
    labeled = np.zeros_like(labeled_ws, dtype=np.uint16)
    new_id = 1
    for iid in range(1, labeled_ws.max() + 1):
        if (labeled_ws == iid).sum() >= MIN_AREA:
            labeled[labeled_ws == iid] = new_id
            new_id += 1
    return labeled


print('=' * 55)
print('PhysGT-CLSM re-run (final manuscript working preset)')
print(f'  sigma_smooth=1.0  threshold=triangle*1.35  close_radius=0')
print(f'  dist_smooth=1.2   min_dist=8 px')
print(f'  MIN_AREA=20 px2')
print('=' * 55)

total = 0
for ct, img_dir in IMG_DIRS.items():
    files = sorted(img_dir.glob('*.tif'))
    for fp in files:
        labeled = segment(fp)
        out = PRED_DIR / fp.name
        tifffile.imwrite(str(out), labeled)
        total += 1
        print(f'  [{total:2d}] {ct}/{fp.name}  n={int(labeled.max())}')

print(f'\nDone. {total} images -> {PRED_DIR}')
