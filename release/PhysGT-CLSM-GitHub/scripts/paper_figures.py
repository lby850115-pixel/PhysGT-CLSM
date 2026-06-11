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
    # Updated to 6 metrics matching Table 1
    metrics = [
        ('mean_area',         'Area (px²)',       'Area'),
        ('mean_aspect_ratio', 'Aspect Ratio',    'Aspect Ratio'),
        ('mean_eccentricity', 'Eccentricity',    'Eccentricity'),
        ('mean_solidity',     'Solidity',        'Solidity'),
        ('mean_tortuosity',   'Tortuosity',      'Tortuosity'),
        ('mean_thickness',    'Thickness (px)',  'Thickness'),
    ]
    ct_list = ['HELA', 'BXPC3', 'MCF7']
    n_models = len(MODELS)

    # Add spacing between models
    spacing = 1.5
    x = np.arange(n_models) * spacing
    w = 0.35
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Cross-Model Morphology Comparison by Cell Type (n=34 images)',
                 fontsize=13, fontweight='bold', y=1.01)

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

        # Highlight PhysGT column with proper spacing
        ax.axvspan(-0.75, 0.75, color=MODEL_COLORS['physegt_clsm'], alpha=0.07, zorder=0)

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
        ax.legend(fontsize=7.5, framealpha=0.7, loc='best')

        # Set x-axis limits
        ax.set_xlim(-1, x[-1] + 1)

    plt.tight_layout()
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig3_morphology_bars.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('fig3_morphology_bars saved (PDF + SVG)')


# =============================================================================
# Fig 4 — Detected count boxplot with broken y-axis (Nellie high, others low)
# =============================================================================
def make_fig4_count(rows):
    import matplotlib.gridspec as gridspec

    ct_list = ['HELA', 'BXPC3', 'MCF7']
    n_models = len(MODELS)

    # Organize data by cell type and model
    count_data = {ct: {m: [float(r['n_mito']) for r in rows
                           if r['model'] == m and r['cell_type'] == ct]
                       for m in MODELS}
                  for ct in ct_list}

    # y-axis break points
    Y_LOW_MAX = 450
    Y_HIGH_MIN = 620
    Y_HIGH_MAX = 1700

    # Create figure with broken axis layout
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle('Detected Mitochondrion Instance Count per Model and Cell Type (n=34 images)',
                 fontsize=13, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2.8], hspace=0.06, wspace=0.28,
                           left=0.07, right=0.98, top=0.91, bottom=0.18)

    positions = np.arange(n_models)
    nellie_idx = MODELS.index('nellie')

    def draw_boxplot(ax, data, positions):
        bp = ax.boxplot(data, positions=positions, patch_artist=True, notch=False, widths=0.58,
                        boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5), medianprops=dict(linewidth=2.2, color='black'),
                        flierprops=dict(marker='o', markersize=3.5, alpha=0.5, linestyle='none'))
        for patch, model in zip(bp['boxes'], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.82)
            patch.set_edgecolor('black')
        bp['boxes'][0].set_linewidth(2.5)
        bp['boxes'][0].set_edgecolor(MODEL_COLORS['physegt_clsm'])
        return bp

    def break_marks(ax_upper, ax_lower):
        d = 0.017
        kw = dict(transform=ax_upper.transAxes, color='k', linewidth=1.3, clip_on=False, zorder=10)
        ax_upper.plot((-d, +d), (-d*1.8, +d*1.8), **kw)
        ax_upper.plot((1-d, 1+d), (-d*1.8, +d*1.8), **kw)
        kw['transform'] = ax_lower.transAxes
        ax_lower.plot((-d, +d), (1-d*1.8, 1+d*1.8), **kw)
        ax_lower.plot((1-d, 1+d), (1-d*1.8, 1+d*1.8), **kw)

    for col, ct in enumerate(ct_list):
        ax_hi = fig.add_subplot(gs[0, col])
        ax_lo = fig.add_subplot(gs[1, col])

        data = [count_data[ct][m] for m in MODELS]
        draw_boxplot(ax_hi, data, positions)
        draw_boxplot(ax_lo, data, positions)

        ax_hi.set_ylim(Y_HIGH_MIN, Y_HIGH_MAX)
        ax_lo.set_ylim(0, Y_LOW_MAX)

        ax_hi.spines['bottom'].set_visible(False)
        ax_lo.spines['top'].set_visible(False)
        ax_hi.tick_params(axis='x', bottom=False, labelbottom=False)
        break_marks(ax_hi, ax_lo)

        # Background highlights
        ax_hi.axvspan(-0.5, 0.5, color=MODEL_COLORS['physegt_clsm'], alpha=0.06, zorder=0)
        ax_lo.axvspan(-0.5, 0.5, color=MODEL_COLORS['physegt_clsm'], alpha=0.06, zorder=0)
        ax_hi.axvspan(nellie_idx-0.5, nellie_idx+0.5, color=MODEL_COLORS['nellie'], alpha=0.08, zorder=0)
        ax_lo.axvspan(nellie_idx-0.5, nellie_idx+0.5, color=MODEL_COLORS['nellie'], alpha=0.08, zorder=0)

        for ax in (ax_hi, ax_lo):
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax_lo.set_xticks(positions)
        ax_lo.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=30, ha='right', fontsize=9)
        for lbl, model in zip(ax_lo.get_xticklabels(), MODELS):
            if model in ('physegt_clsm', 'nellie'):
                lbl.set_color(MODEL_COLORS[model])
                lbl.set_fontweight('bold')

        ax_hi.set_title(CT_LABELS[ct], fontweight='bold', fontsize=11, pad=6)
        if col == 0:
            ax_lo.set_ylabel('Instance count per image', fontsize=10)

        # Nellie median annotation
        nellie_vals = count_data[ct]['nellie']
        if nellie_vals:
            nellie_med = int(np.median(nellie_vals))
            nellie_mean = int(np.mean(nellie_vals))
            ax_hi.text(nellie_idx, Y_HIGH_MAX-60, f'med={nellie_med}\nmean={nellie_mean}',
                      ha='center', va='top', fontsize=7.5, color=MODEL_COLORS['nellie'],
                      fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

        if col == 0:
            ax_lo.text(-0.14, 1.02, '// break //', transform=ax_lo.transAxes,
                      fontsize=7, color='#777777', va='bottom')

        n = len(count_data[ct][MODELS[0]])
        ax_lo.text(0.02, 0.97, f'n={n} images', transform=ax_lo.transAxes, fontsize=8, va='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'fig4_count_boxplot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('fig4_count_boxplot saved (PDF + SVG, broken-axis)')


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
