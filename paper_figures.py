"""
paper_figures.py
================
Generate publication-quality vector figures (PDF + SVG) for the PhysGT-CLSM paper.

Story:
  1. PhysGT-CLSM is proposed to solve the GT annotation bottleneck for CLSM mitochondria.
  2. Visual comparison: PhysGT-CLSM segmentation overlays vs 5 baseline models (6 images).
  3. Morphology quantification: PhysGT-CLSM vs 5 models across 34 images × 3 cell types.

Outputs (figures/paper/):
  fig1_overlay_grid.pdf/.svg   — 6×6 segmentation overlay grid (raw + 6 models)
  fig2_morphology_violin.pdf/.svg — violin plots: 6 key metrics × 6 models
  fig3_morphology_bars.pdf/.svg  — bar chart: mean ± SD per cell type per model
  fig4_count_boxplot.pdf/.svg    — detected mito count per model × cell type
"""

import csv, warnings
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from pathlib import Path
from skimage.segmentation import find_boundaries

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parent
OUT_DIR  = ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config ──────────────────────────────────────────────────────────────
MODELS = ['physegt_clsm', 'cellpose', 'mitosegnet', 'modl', 'nellie', 'mitometer']
MODEL_LABELS = {
    'physegt_clsm': 'PhysGT-CLSM\n(Ours)',
    'cellpose':     'Cellpose',
    'mitosegnet':   'MitoSegNet',
    'modl':         'MoDL',
    'nellie':       'Nellie',
    'mitometer':    'Mitometer',
}
MODEL_COLORS = {
    'physegt_clsm': '#E63946',   # red — highlight ours
    'cellpose':     '#457B9D',
    'mitosegnet':   '#2A9D8F',
    'modl':         '#E9C46A',
    'nellie':       '#F4A261',
    'mitometer':    '#6D6875',
}

CT_LABELS = {'HELA': 'HeLa', 'BXPC3': 'BxPC-3', 'MCF7': 'MCF-7'}
CT_COLORS = {'HELA': '#4E79A7', 'BXPC3': '#F28E2B', 'MCF7': '#59A14F'}

MITO_CHANNEL = 1   # green channel index in (H,W,C) TIFs

# ── 6 reference images (MITO DATA_1) ─────────────────────────────────────────
REF_DIR   = ROOT / 'MITO DATA_1'
PRED_DIR6 = ROOT / 'predictions'
REF_NAMES = {
    'Series086': 'S086', 'Series090': 'S090', 'Series095': 'S095',
    'Series099': 'S099', 'Series103': 'S103', 'Series109': 'S109',
}

def short_name(stem):
    for k, v in REF_NAMES.items():
        if stem.startswith(k):
            return v
    return stem[:6]

def load_channel(fp):
    img = tifffile.imread(str(fp))
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[0] <= 4:
            return img[MITO_CHANNEL].astype(np.float32)
        return img[:, :, MITO_CHANNEL].astype(np.float32)
    return img[..., MITO_CHANNEL].astype(np.float32)

def norm01(arr):
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

def label_to_rgba(labels, alpha=0.65):
    n = int(labels.max())
    if n == 0:
        return np.zeros((*labels.shape, 4), dtype=np.float32)
    cmap = cm.get_cmap('tab20', max(n, 1))
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for iid in range(1, n + 1):
        mask = labels == iid
        c = cmap(iid % 20)
        rgba[mask] = [c[0], c[1], c[2], alpha]
    return rgba

# ── Load morphology data ──────────────────────────────────────────────────────
def load_morphology():
    fp = ROOT / 'results' / 'morphology_34' / 'all_models_summary.csv'
    rows = list(csv.DictReader(open(fp, encoding='utf-8')))
    return rows

# =============================================================================
# Fig 1 — Segmentation overlay grid (6 images × 7 columns: raw + 6 models)
# =============================================================================
def make_fig1_overlay():
    ref_files = sorted(REF_DIR.glob('*.tif'))
    if not ref_files:
        print('[WARN] No TIF files in MITO DATA_1/, skipping fig1')
        return

    n_images = len(ref_files)
    n_cols   = 1 + len(MODELS)   # raw + 6 models
    fig, axes = plt.subplots(n_images, n_cols,
                             figsize=(n_cols * 2.2, n_images * 2.2),
                             dpi=150)
    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Raw'] + [MODEL_LABELS[m].replace('\n', ' ') for m in MODELS]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, fontweight='bold',
                             color=MODEL_COLORS.get(MODELS[j-1], 'black') if j > 0 else 'black')

    for i, fp in enumerate(ref_files):
        sname = short_name(fp.stem)
        raw   = norm01(load_channel(fp))

        # Row label
        axes[i, 0].set_ylabel(sname, fontsize=8, rotation=0, labelpad=28, va='center')

        # Col 0: raw
        axes[i, 0].imshow(raw, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')

        # Cols 1-6: model overlays
        for j, model in enumerate(MODELS, start=1):
            pred_fp = PRED_DIR6 / model / fp.name
            ax = axes[i, j]
            ax.imshow(raw, cmap='gray', vmin=0, vmax=1)
            if pred_fp.exists():
                labels = tifffile.imread(str(pred_fp)).astype(np.uint32)
                rgba   = label_to_rgba(labels, alpha=0.55)
                ax.imshow(rgba)
                n_inst = int(labels.max())
                ax.text(0.02, 0.97, f'n={n_inst}', transform=ax.transAxes,
                        fontsize=6, color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5, lw=0))
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
            ax.axis('off')

    plt.tight_layout(pad=0.3)
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig1_overlay_grid.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('fig1_overlay_grid saved (PDF + SVG)')


# =============================================================================
# Fig 2 — Violin plots: 6 key morphology metrics × 6 models (all 34 images)
# =============================================================================
def make_fig2_violin(rows):
    metrics = [
        ('mean_area',         'Area (px²)',      'Mitochondrion Area'),
        ('mean_aspect_ratio', 'Aspect Ratio',    'Aspect Ratio'),
        ('mean_eccentricity', 'Eccentricity',    'Eccentricity'),
        ('mean_solidity',     'Solidity',        'Solidity'),
        ('mean_tortuosity',   'Tortuosity',      'Tortuosity'),
        ('mean_thickness',    'Thickness (px)',  'Thickness'),
    ]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('Morphological Metrics Across Models (34 images)',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax, (col, ylabel, title) in zip(axes.flat, metrics):
        data   = []
        colors = []
        for model in MODELS:
            vals = [float(r[col]) for r in rows
                    if r['model'] == model and r.get(col, '') != '']
            data.append(np.array(vals) if vals else np.array([0.0]))
            colors.append(MODEL_COLORS[model])

        parts = ax.violinplot(data, showmedians=True, showextrema=False,
                              widths=0.7)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.6)
            pc.set_alpha(0.8)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.2)

        # Scatter jitter overlay
        for k, (model, d) in enumerate(zip(MODELS, data), start=1):
            jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(d))
            ax.scatter(np.full(len(d), k) + jitter, d,
                       s=12, color=MODEL_COLORS[model], alpha=0.5,
                       edgecolors='none', zorder=3)

        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels([MODEL_LABELS[m].replace('\n', '\n') for m in MODELS],
                           fontsize=7, rotation=30, ha='right')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        # Highlight "ours" x-tick
        ax.get_xticklabels()[0].set_color(MODEL_COLORS['physegt_clsm'])
        ax.get_xticklabels()[0].set_fontweight('bold')

    plt.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig2_morphology_violin.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('fig2_morphology_violin saved (PDF + SVG)')


# =============================================================================
# Fig 3 — Bar chart: mean ± SD per model, grouped by cell type (4 key metrics)
# =============================================================================
def make_fig3_bars(rows):
    metrics = [
        ('mean_area',         'Area (px²)',    'Mitochondrion Area'),
        ('mean_aspect_ratio', 'Aspect Ratio',  'Aspect Ratio'),
        ('mean_tortuosity',   'Tortuosity',    'Tortuosity'),
        ('mean_eccentricity', 'Eccentricity',  'Eccentricity'),
    ]
    ct_list = ['HELA', 'BXPC3', 'MCF7']
    n_models = len(MODELS)
    x = np.arange(n_models)
    w = 0.22
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Cross-Model Morphology Comparison by Cell Type',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax, (col, ylabel, title) in zip(axes.flat, metrics):
        for ct, offset in zip(ct_list, offsets):
            means, stds = [], []
            for model in MODELS:
                vals = [float(r[col]) for r in rows
                        if r['model'] == model and r['cell_type'] == ct
                        and r.get(col, '') != '']
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals)   if vals else 0)
            bars = ax.bar(x + offset, means, w, yerr=stds,
                          label=CT_LABELS[ct], color=CT_COLORS[ct],
                          edgecolor='black', linewidth=0.5,
                          capsize=3, alpha=0.85, error_kw={'linewidth': 0.8})

        # Highlight PhysGT column with a box
        ax.axvspan(-0.5, 0.5, color=MODEL_COLORS['physegt_clsm'], alpha=0.07, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m].replace('\n', '\n') for m in MODELS],
                           fontsize=8, rotation=25, ha='right')
        ax.get_xticklabels()[0].set_color(MODEL_COLORS['physegt_clsm'])
        ax.get_xticklabels()[0].set_fontweight('bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.legend(fontsize=8, framealpha=0.7)

    plt.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig3_morphology_bars.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('fig3_morphology_bars saved (PDF + SVG)')


# =============================================================================
# Fig 4 — Detected count boxplot: model × cell type
# =============================================================================
def make_fig4_count(rows):
    ct_list = ['HELA', 'BXPC3', 'MCF7']
    n_models = len(MODELS)
    x = np.arange(n_models)
    w = 0.22
    offsets = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Detected Mitochondrion Count per Model and Cell Type',
                 fontsize=12, fontweight='bold')

    for ct, offset in zip(ct_list, offsets):
        means, stds = [], []
        for model in MODELS:
            vals = [float(r['n_mito']) for r in rows
                    if r['model'] == model and r['cell_type'] == ct]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)   if vals else 0)
        ax.bar(x + offset, means, w, yerr=stds,
               label=CT_LABELS[ct], color=CT_COLORS[ct],
               edgecolor='black', linewidth=0.5,
               capsize=3, alpha=0.85, error_kw={'linewidth': 0.8})

    ax.axvspan(-0.5, 0.5, color=MODEL_COLORS['physegt_clsm'], alpha=0.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m].replace('\n', '\n') for m in MODELS],
                       fontsize=9, rotation=25, ha='right')
    ax.get_xticklabels()[0].set_color(MODEL_COLORS['physegt_clsm'])
    ax.get_xticklabels()[0].set_fontweight('bold')
    ax.set_ylabel('Detected count (mean ± SD)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(fontsize=9, framealpha=0.7)

    plt.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig4_count_boxplot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('fig4_count_boxplot saved (PDF + SVG)')


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('Loading morphology data...')
    rows = load_morphology()
    print(f'  {len(rows)} rows loaded')

    from collections import Counter
    model_counts = Counter(r['model'] for r in rows)
    ct_counts    = Counter(r['cell_type'] for r in rows)
    print(f'  Models: {dict(model_counts)}')
    print(f'  Cell types: {dict(ct_counts)}')

    print('\nGenerating figures...')
    make_fig1_overlay()
    make_fig2_violin(rows)
    make_fig3_bars(rows)
    make_fig4_count(rows)

    print(f'\nAll figures saved to: {OUT_DIR}')
    print('Files:')
    for f in sorted(OUT_DIR.iterdir()):
        print(f'  {f.name}')
