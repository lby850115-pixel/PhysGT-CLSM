"""
stats_significance.py
=====================
Kruskal-Wallis + Dunn post-hoc tests on 6 morphology metrics across 6 models.
Outputs:
  results/stats_kruskal.csv        — per-metric H-statistic and p-value
  results/stats_dunn_<metric>.csv  — pairwise Dunn p-values (BH-corrected)
  results/stats_significance_table.csv — compact table with significance stars
  figures/paper/figS2_stats_heatmap.pdf/.svg — heatmap of pairwise significance
"""

import csv, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')

ROOT    = Path(__file__).resolve().parent
OUT_RES = ROOT / 'results'
OUT_FIG = ROOT / 'figures' / 'paper'
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
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
    'physegt_clsm': '#E63946',
    'cellpose':     '#457B9D',
    'mitosegnet':   '#2A9D8F',
    'modl':         '#E9C46A',
    'nellie':       '#F4A261',
    'mitometer':    '#6D6875',
}

METRICS = [
    ('mean_area',         'Area (px2)'),
    ('mean_aspect_ratio', 'Aspect Ratio'),
    ('mean_eccentricity', 'Eccentricity'),
    ('mean_solidity',     'Solidity'),
    ('mean_tortuosity',   'Tortuosity'),
    ('mean_thickness',    'Thickness (px)'),
]

# ── Load data ─────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(
    OUT_RES / 'morphology_34' / 'all_models_summary.csv', encoding='utf-8')))

def get_vals(model, metric):
    return [float(r[metric]) for r in rows
            if r['model'] == model and r.get(metric, '') != '']

# ── BH correction ─────────────────────────────────────────────────────────────
def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
    adjusted = np.minimum(1.0, pvals * n / ranks)
    # enforce monotonicity (cumulative min from largest rank)
    for i in range(n-2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i+1]])
    return adjusted

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── Kruskal-Wallis ────────────────────────────────────────────────────────────
print('=' * 65)
print('KRUSKAL-WALLIS TEST  (6 models × 6 metrics, 34 images each)')
print('=' * 65)

kw_rows = []
for col, label in METRICS:
    groups = [get_vals(m, col) for m in MODELS]
    groups = [g for g in groups if len(g) > 0]
    H, p = stats.kruskal(*groups)
    kw_rows.append({'metric': col, 'label': label,
                    'H': round(H, 3), 'p': p,
                    'p_str': f'{p:.2e}', 'sig': stars(p)})
    print(f'  {label:<22}: H={H:.2f}  p={p:.2e}  {stars(p)}')

with open(OUT_RES / 'stats_kruskal.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['metric','label','H','p','p_str','sig'])
    w.writeheader(); w.writerows(kw_rows)

# ── Dunn post-hoc (BH-corrected) ─────────────────────────────────────────────
print('\n' + '=' * 65)
print('DUNN POST-HOC  (pairwise, BH-corrected)')
print('=' * 65)

pairs = list(combinations(MODELS, 2))
# store: dunn_results[metric][pair] = adjusted_p
dunn_results = {}

for col, label in METRICS:
    print(f'\n  {label}:')
    groups = {m: get_vals(m, col) for m in MODELS}

    # Dunn test via rank-sum z-scores (manual implementation, no scikit-posthocs needed)
    all_vals = []
    group_ids = []
    for i, m in enumerate(MODELS):
        all_vals.extend(groups[m])
        group_ids.extend([i] * len(groups[m]))
    all_vals = np.array(all_vals)
    group_ids = np.array(group_ids)
    N = len(all_vals)

    # Ranks (average for ties)
    ranks = stats.rankdata(all_vals)

    # Tie correction factor
    _, tie_counts = np.unique(all_vals, return_counts=True)
    tie_factor = np.sum(tie_counts**3 - tie_counts) / (12 * (N - 1))

    raw_pvals = []
    for (m1, m2) in pairs:
        i1 = MODELS.index(m1); i2 = MODELS.index(m2)
        r1 = ranks[group_ids == i1]; r2 = ranks[group_ids == i2]
        n1 = len(r1); n2 = len(r2)
        mean_r1 = r1.mean(); mean_r2 = r2.mean()
        se = np.sqrt((N*(N+1)/12 - tie_factor) * (1/n1 + 1/n2))
        z = (mean_r1 - mean_r2) / se if se > 0 else 0.0
        p_raw = 2 * stats.norm.sf(abs(z))
        raw_pvals.append(p_raw)

    adj_pvals = bh_correct(np.array(raw_pvals))
    dunn_results[col] = {}
    for (m1, m2), p_adj in zip(pairs, adj_pvals):
        dunn_results[col][(m1, m2)] = p_adj
        dunn_results[col][(m2, m1)] = p_adj
        if p_adj < 0.05:
            print(f'    {MODEL_LABELS[m1].replace(chr(10)," "):<22} vs '
                  f'{MODEL_LABELS[m2].replace(chr(10)," "):<14}: '
                  f'p={p_adj:.3e}  {stars(p_adj)}')

    # Save pairwise CSV
    csv_rows = []
    for (m1, m2), p_adj in zip(pairs, adj_pvals):
        csv_rows.append({'model_a': m1, 'model_b': m2,
                         'p_adj': round(p_adj, 6), 'sig': stars(p_adj)})
    with open(OUT_RES / f'stats_dunn_{col}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['model_a','model_b','p_adj','sig'])
        w.writeheader(); w.writerows(csv_rows)

# ── Compact significance table (models as rows, metrics as cols) ──────────────
# For each metric: show PhysGT-CLSM vs every other model
print('\n' + '=' * 65)
print('PHYSGT-CLSM vs OTHERS  (BH-corrected Dunn p-values)')
print('=' * 65)
header = ['comparison'] + [label for _, label in METRICS]
sig_rows = []
for m2 in MODELS[1:]:
    row = {'comparison': f'PhysGT vs {MODEL_LABELS[m2].replace(chr(10)," ")}'}
    for col, label in METRICS:
        p = dunn_results[col].get(('physegt_clsm', m2), 1.0)
        row[label] = f'{stars(p)} ({p:.2e})'
    sig_rows.append(row)
    vals = [row[label] for _, label in METRICS]
    print(f'  {row["comparison"]:<30}: {" | ".join(v[:12] for v in vals)}')

with open(OUT_RES / 'stats_significance_table.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader(); w.writerows(sig_rows)

# ── Figure: pairwise significance heatmap ────────────────────────────────────
n_metrics = len(METRICS)
n_models  = len(MODELS)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Pairwise Significance (Dunn post-hoc, BH-corrected)\nKruskal–Wallis across 6 models × 34 images',
             fontsize=11, fontweight='bold', y=1.01)

for ax, (col, label) in zip(axes.flat, METRICS):
    mat = np.ones((n_models, n_models))
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i != j:
                mat[i, j] = dunn_results[col].get((m1, m2), 1.0)

    # -log10(p), capped at 4
    log_mat = np.clip(-np.log10(mat + 1e-10), 0, 4)
    np.fill_diagonal(log_mat, 0)

    im = ax.imshow(log_mat, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')

    tick_labels = [MODEL_LABELS[m].replace('\n', '\n') for m in MODELS]
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=35, ha='right')
    ax.set_yticklabels(tick_labels, fontsize=6.5)

    # Annotate cells with stars
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            p = dunn_results[col].get((MODELS[i], MODELS[j]), 1.0)
            s = stars(p)
            color = 'white' if log_mat[i, j] > 2.5 else 'black'
            ax.text(j, i, s, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    # Highlight PhysGT row/col
    for k in range(n_models):
        ax.add_patch(plt.Rectangle((k-0.5, -0.5), 1, 1,
                     fill=False, edgecolor='#E63946', lw=1.5 if k == 0 else 0))
        ax.add_patch(plt.Rectangle((-0.5, k-0.5), 1, 1,
                     fill=False, edgecolor='#E63946', lw=1.5 if k == 0 else 0))

    ax.set_title(label, fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='-log10(p)', shrink=0.8)

# Legend
fig.text(0.5, -0.02,
         '*** p<0.001   ** p<0.01   * p<0.05   ns p>=0.05  |  '
         'Red border = PhysGT-CLSM row/column',
         ha='center', fontsize=8, color='#444')

plt.tight_layout()
for ext in ('pdf', 'svg'):
    fig.savefig(OUT_FIG / f'figS2_stats_heatmap.{ext}', bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'\nfigS2_stats_heatmap saved to {OUT_FIG}')

print('\nAll outputs:')
for f in sorted(OUT_RES.glob('stats_*.csv')):
    print(f'  {f.name}')
