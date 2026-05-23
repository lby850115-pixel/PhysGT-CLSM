"""
08_morphology_34.py

Compute 22 morphology metrics for all 34 CLSM images × all available models
from predictions_34/. Mirrors the metric set in 03_morphology_modl.py exactly,
adding cell_type and model columns.

Input
-----
  predictions_34/{model}/{stem}.tif   uint16 instance-label TIFs

Output
------
  results/morphology_34/{model}_per_instance.csv
  results/morphology_34/{model}_summary.csv
  results/morphology_34/all_models_summary.csv   (combined, for cross-model stats)
  figures/morphology_34/{model}_*.png

Usage
-----
  python 08_morphology_34.py
  python 08_morphology_34.py --models nellie mitometer
"""

from __future__ import annotations
import argparse, csv, warnings
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label as ndlabel, distance_transform_edt
from skimage.morphology import thin
from skimage.measure import regionprops

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parent
PRED_DIR = ROOT / 'predictions_34'
OUT_RES  = ROOT / 'results'  / 'morphology_34'
OUT_FIG  = ROOT / 'figures'  / 'morphology_34'
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

ALL_MODELS = ['cellpose', 'mitosegnet', 'modl', 'nellie', 'mitometer', 'physegt_clsm']
MODEL_LABELS = {
    'cellpose':     'Cellpose',
    'mitosegnet':   'MitoSegNet',
    'modl':         'MoDL',
    'nellie':       'Nellie',
    'mitometer':    'Mitometer',
    'physegt_clsm': 'PhysGT-CLSM',
}

IMG_DIRS = {
    'HELA':  ROOT / 'MITO DATA' / 'HELA',
    'BXPC3': ROOT / 'MITO DATA' / 'BXPC-3',
    'MCF7':  ROOT / 'MITO DATA' / 'MCF-7',
}
CT_LABELS = {'HELA': 'HELA', 'BXPC3': 'BxPC-3', 'MCF7': 'MCF-7'}

METRICS_NUM = [
    'area', 'solidity', 'extent', 'major_axis', 'minor_axis', 'aspect_ratio', 'thickness',
    'inertia_min', 'inertia_mid', 'inertia_max',
    'eccentricity', 'perimeter', 'form_factor', 'roundness',
    'skel_length', 'n_branch_points', 'n_branches', 'tortuosity',
    'total_branch_length', 'mean_branch_length', 'median_branch_length', 'std_branch_length',
]

# ── Branch-point detection (identical to 03_morphology_modl.py) ───────────────

_VALID_BP_SET = {tuple(v) for v in [
    [0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],[0,1,0,0,1,0,1,0],
    [0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],[0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],
    [0,1,0,0,0,1,0,1],[0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
    [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],[1,0,0,0,1,0,1,0],
    [1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],[1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],
    [1,0,1,1,0,0,1,0],[1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
    [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],[0,0,1,0,1,0,1,1],
    [1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],[1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],
    [0,1,0,0,1,0,1,1],[0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
    [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],[1,0,1,1,0,1,0,0],
]}


def get_branch_points(skel_bool):
    img = skel_bool.astype(np.uint8)
    # Extract 8-neighbour patterns for all skeleton pixels at once
    ys, xs = np.where(img[1:-1, 1:-1])
    ys += 1; xs += 1
    if len(ys) == 0:
        return []
    # neighbours in order: W, NW, N, NE, E, SE, S, SW
    nb = np.stack([img[ys, xs-1], img[ys-1, xs-1], img[ys-1, xs], img[ys-1, xs+1],
                   img[ys, xs+1], img[ys+1, xs+1], img[ys+1, xs], img[ys+1, xs-1]], axis=1)
    pts = [(int(ys[i]), int(xs[i])) for i in range(len(ys))
           if tuple(nb[i].tolist()) in _VALID_BP_SET]
    # Deduplicate: keep points >10px apart
    keep = []
    for p in pts:
        if not any((p[0]-k[0])**2+(p[1]-k[1])**2 < 100 for k in keep):
            keep.append(p)
    return keep


# ── Per-instance metric computation ──────────────────────────────────────────

def instance_metrics(binary_mask, img_name, iid, cell_type, model):
    props  = regionprops(binary_mask.astype(np.uint8))[0]
    area   = props.area
    perim  = props.perimeter
    major  = props.major_axis_length
    minor  = props.minor_axis_length

    eigs = sorted(props.inertia_tensor_eigvals)
    inertia_min = float(eigs[0])
    inertia_mid = float(eigs[min(1, len(eigs)-1)])
    inertia_max = float(eigs[-1])
    aspect_ratio = float(major / minor) if minor > 0 else float(major)
    thickness    = float(distance_transform_edt(binary_mask)[binary_mask].mean()) \
                   if binary_mask.any() else 0.0
    form_factor  = (perim**2 / area) / (4 * np.pi) if area > 0 else 0.0
    roundness    = (4 / np.pi) * area / (major**2)  if major > 0 else 0.0

    # Crop to bounding box before thin() to avoid processing large zero regions
    rows_nz, cols_nz = np.where(binary_mask)
    r0, r1 = rows_nz.min(), rows_nz.max() + 1
    c0, c1 = cols_nz.min(), cols_nz.max() + 1
    crop = binary_mask[r0:r1, c0:c1]

    if crop.sum() > 2000:
        # Too large for thin() — estimate from major axis
        skel_px   = int(major)
        bp_list   = []
        n_branches = 1
        branch_lengths = [float(major)]
    else:
        skel_crop = thin(crop)
        skel_bool_crop = skel_crop > 0
        skel_px   = int(np.count_nonzero(skel_bool_crop))
        bp_list   = get_branch_points(skel_bool_crop)
        bp_img = np.zeros_like(crop, dtype=np.uint8)
        for (r, c) in bp_list:
            bp_img[max(0,r-1):r+2, max(0,c-1):c+2] = 1
        seg_labels, n_branches = ndlabel(skel_bool_crop & (bp_img == 0))
        branch_lengths = []
        if n_branches > 0:
            for sid in range(1, n_branches + 1):
                branch_lengths.append(int(np.sum(seg_labels == sid)) + 4)
            if len(branch_lengths) == 1:
                branch_lengths[0] = float(major)
        else:
            branch_lengths = [float(major)]
            n_branches = 1

    tortuosity = float(skel_px / major) if major > 0 else 1.0

    return {
        'image': img_name, 'cell_type': cell_type, 'model': model, 'iid': iid,
        'area': float(area), 'solidity': float(props.solidity),
        'extent': float(props.extent), 'major_axis': float(major),
        'minor_axis': float(minor), 'aspect_ratio': aspect_ratio,
        'thickness': thickness, 'inertia_min': inertia_min,
        'inertia_mid': inertia_mid, 'inertia_max': inertia_max,
        'eccentricity': float(props.eccentricity), 'perimeter': float(perim),
        'form_factor': float(form_factor), 'roundness': float(roundness),
        'skel_length': skel_px, 'n_branch_points': len(bp_list),
        'n_branches': n_branches, 'tortuosity': tortuosity,
        'total_branch_length': float(np.sum(branch_lengths)),
        'mean_branch_length':  float(np.mean(branch_lengths)),
        'median_branch_length': float(np.median(branch_lengths)),
        'std_branch_length':   float(np.std(branch_lengths)),
    }


# ── Process one model ─────────────────────────────────────────────────────────

def process_model(model: str, image_list: list[dict]):
    pred_dir = PRED_DIR / model
    if not pred_dir.exists():
        print(f'[SKIP] predictions_34/{model}/ not found.')
        return [], []

    all_rows, agg_rows = [], []

    print(f'\n{"="*60}')
    print(f'Model: {MODEL_LABELS[model]}')
    print(f'{"="*60}')
    print(f'{"Image":<35} {"CT":<7} {"N":>5} {"AreaMed":>9} {"Tort":>7} {"AR":>7}')
    print('-'*70)

    for item in image_list:
        stem      = item['stem']
        cell_type = item['cell_type']
        pred_fp   = pred_dir / f'{stem}.tif'

        if not pred_fp.exists():
            print(f'  [miss] {stem}')
            continue

        inst  = tifffile.imread(pred_fp).astype(np.uint32)
        n_inst = int(inst.max())
        per_inst = []

        ids = list(range(1, n_inst + 1))
        if len(ids) > 300:
            rng = np.random.default_rng(42)
            ids = sorted(rng.choice(ids, 300, replace=False).tolist())

        for iid in ids:
            bm = (inst == iid)
            if bm.sum() <= 16:
                continue
            try:
                m = instance_metrics(bm, stem, iid, cell_type, model)
                per_inst.append(m)
                all_rows.append(m)
            except Exception as e:
                pass  # skip degenerate instances silently

        if not per_inst:
            continue

        rec = {'image': stem, 'cell_type': cell_type, 'model': model,
               'n_mito': len(per_inst)}
        for col in METRICS_NUM:
            vals = np.array([r[col] for r in per_inst], dtype=float)
            rec[f'mean_{col}']   = float(vals.mean())
            rec[f'median_{col}'] = float(np.median(vals))
            rec[f'std_{col}']    = float(vals.std())
        agg_rows.append(rec)

        areas_list = [r['area'] for r in per_inst]
        ct_label   = CT_LABELS.get(cell_type, cell_type)
        print(f'  {stem:<35} {ct_label:<7} {len(per_inst):>5} '
              f'{int(np.median(areas_list)):>9} '
              f'{rec["mean_tortuosity"]:>7.3f} '
              f'{rec["mean_aspect_ratio"]:>7.2f}')

    return all_rows, agg_rows


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ── Per-model summary figure ──────────────────────────────────────────────────

def make_model_figure(agg_rows, model):
    if not agg_rows:
        return
    ct_colors = {'HELA': 'steelblue', 'BXPC3': 'darkorange', 'MCF7': 'seagreen'}
    metrics_to_plot = [
        ('area',         'Area (px)',      'Area'),
        ('eccentricity', 'Eccentricity',   'Eccentricity'),
        ('solidity',     'Solidity',       'Solidity'),
        ('aspect_ratio', 'Aspect Ratio',   'Aspect Ratio'),
        ('tortuosity',   'Tortuosity',     'Tortuosity'),
        ('thickness',    'Thickness (px)', 'Thickness'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'{MODEL_LABELS[model]} — Morphology across 34 images',
                 fontsize=13, fontweight='bold')
    for ax, (col, ylabel, title) in zip(axes.flat, metrics_to_plot):
        by_ct = {ct: [] for ct in ['HELA', 'BXPC3', 'MCF7']}
        for r in agg_rows:
            by_ct[r['cell_type']].append(r[f'mean_{col}'])
        x    = np.arange(3)
        cts  = ['HELA', 'BXPC3', 'MCF7']
        means = [np.mean(by_ct[ct]) if by_ct[ct] else 0 for ct in cts]
        stds  = [np.std(by_ct[ct])  if by_ct[ct] else 0 for ct in cts]
        bars  = ax.bar(x, means, color=[ct_colors[ct] for ct in cts],
                       edgecolor='black', linewidth=0.7, alpha=0.8, width=0.5)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([CT_LABELS[ct] for ct in cts], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_FIG / f'{model}_celltype_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure → {fig_path}')


# ── Cross-model comparison figure ─────────────────────────────────────────────

def make_crossmodel_figure(combined_agg):
    """Bar chart: key metrics by model, grouped by cell type."""
    key_metrics = [
        ('mean_area',         'Area (px)',    'Mean Area'),
        ('mean_aspect_ratio', 'Aspect Ratio', 'Mean Aspect Ratio'),
        ('mean_tortuosity',   'Tortuosity',   'Mean Tortuosity'),
        ('mean_eccentricity', 'Eccentricity', 'Mean Eccentricity'),
    ]
    models_present = [m for m in ALL_MODELS if any(r['model'] == m for r in combined_agg)]
    n_models = len(models_present)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-model morphology comparison (all 34 images)',
                 fontsize=13, fontweight='bold')
    colors = plt.cm.get_cmap('tab10', n_models)
    x = np.arange(n_models)
    w = 0.25
    ct_list = ['HELA', 'BXPC3', 'MCF7']
    ct_offsets = [-w, 0, w]

    for ax, (col, ylabel, title) in zip(axes.flat, key_metrics):
        for ct, offset in zip(ct_list, ct_offsets):
            means, stds = [], []
            for model in models_present:
                rows = [r for r in combined_agg
                        if r['model'] == model and r['cell_type'] == ct]
                vals = [r[col] for r in rows if r.get(col) is not None]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals)   if vals else 0)
            bars = ax.bar(x + offset, means, w, yerr=stds, capsize=3,
                          label=CT_LABELS[ct], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models_present],
                           rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_FIG / 'crossmodel_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nCross-model figure → {fig_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None,
                        choices=ALL_MODELS, help='Models to process (default: all available)')
    args = parser.parse_args()

    # Build image list with cell type
    image_list = []
    for cell_type, img_dir in IMG_DIRS.items():
        if not img_dir.exists():
            continue
        for fp in sorted(img_dir.glob('*.tif')):
            image_list.append({'stem': fp.stem, 'cell_type': cell_type})
    print(f'Total images: {len(image_list)}')

    models_to_run = args.models or [
        m for m in ALL_MODELS if (PRED_DIR / m).exists()
    ]
    print(f'Models: {models_to_run}')

    combined_per_inst = []
    combined_agg      = []

    for model in models_to_run:
        all_rows, agg_rows = process_model(model, image_list)
        if not all_rows:
            continue

        # Save per-model CSVs
        save_csv(all_rows, OUT_RES / f'{model}_per_instance.csv')
        save_csv(agg_rows, OUT_RES / f'{model}_summary.csv')
        print(f'  CSVs → results/morphology_34/{model}_*.csv')

        make_model_figure(agg_rows, model)

        combined_per_inst.extend(all_rows)
        combined_agg.extend(agg_rows)

    # Combined summary CSV
    if combined_agg:
        save_csv(combined_agg, OUT_RES / 'all_models_summary.csv')
        print(f'\nCombined summary → {OUT_RES}/all_models_summary.csv')
        make_crossmodel_figure(combined_agg)

    print(f'\nAll outputs → {OUT_RES}')
    print(f'Figures     → {OUT_FIG}')


if __name__ == '__main__':
    main()
