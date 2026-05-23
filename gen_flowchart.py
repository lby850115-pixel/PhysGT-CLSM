"""
gen_flowchart.py
================
Generates a two-track pipeline flowchart for PhysGT-CLSM (SCI submission quality).
Output: figures/paper/fig_pipeline.svg  +  fig_pipeline.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures' / 'paper'
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT  = '#1565C0'   # dark blue  — input data
C_PREP   = '#0288D1'   # light blue — preprocessing
C_SIM    = '#2E7D32'   # dark green — simulation track
C_SEG    = '#E65100'   # dark orange — segmentation track
C_ANAL   = '#6A1B9A'   # purple     — analysis
C_VAL    = '#C62828'   # red        — validation / output
C_BG_SIM = '#F1F8E9'   # pale green background for sim track
C_BG_SEG = '#FFF3E0'   # pale orange background for seg track
C_ARROW  = '#37474F'   # dark grey arrows

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helper functions ──────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, title, subtitle='', color='#1565C0', fontsize_title=8.5,
        fontsize_sub=6.8, text_color='white', radius=0.25):
    """Draw a rounded rectangle with title + optional subtitle."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle=f'round,pad={radius}',
                          facecolor=color, edgecolor='white',
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    if subtitle:
        ax.text(cx, cy + h*0.13, title, ha='center', va='center',
                fontsize=fontsize_title, fontweight='bold', color=text_color,
                zorder=4, wrap=True)
        ax.text(cx, cy - h*0.22, subtitle, ha='center', va='center',
                fontsize=fontsize_sub, color=text_color, alpha=0.88,
                style='italic', zorder=4)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=fontsize_title, fontweight='bold', color=text_color,
                zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.6, rad=0.0):
    """Draw a straight or curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=5)

def label(ax, x, y, text, fontsize=7.2, color='#37474F', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            color=color, zorder=6)

# ── Background panels ─────────────────────────────────────────────────────────
# Left panel: Real image pipeline
left_bg = FancyBboxPatch((0.3, 0.4), 8.0, 10.2,
                         boxstyle='round,pad=0.2',
                         facecolor=C_BG_SEG, edgecolor='#E65100',
                         linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)
ax.add_patch(left_bg)
ax.text(4.3, 10.75, 'Track A  —  Real Image Segmentation Pipeline',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#E65100', zorder=6)

# Right panel: Simulation track
right_bg = FancyBboxPatch((9.7, 0.4), 7.9, 10.2,
                          boxstyle='round,pad=0.2',
                          facecolor=C_BG_SIM, edgecolor='#2E7D32',
                          linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)
ax.add_patch(right_bg)
ax.text(13.65, 10.75, 'Track B  —  Synthetic Self-Validation',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#2E7D32', zorder=6)

# ── TRACK A: Real image pipeline (left column) ────────────────────────────────
BW, BH = 3.6, 0.88   # box width / height
LX = 4.3             # left column centre x

# Step 1: Input
box(ax, LX, 10.0, BW, BH,
    'Input: 34 CLSM Images',
    'HeLa (n=20) · BxPC-3 (n=6) · MCF-7 (n=8)\n1024×1024 px, 120.25 nm/px',
    color=C_INPUT)

arrow(ax, LX, 10.0 - BH/2, LX, 8.95 + BH/2)

# Step 2: Preprocessing
box(ax, LX, 8.95, BW, BH,
    'Preprocessing',
    '1%–99% clip · Gaussian σ=1.0 px',
    color=C_PREP)

arrow(ax, LX, 8.95 - BH/2, LX, 7.9 + BH/2)

# Step 3: Thresholding
box(ax, LX, 7.9, BW, BH,
    'Triangle Threshold + Binary Closing',
    'Sparse foreground (<5% area) · disk(1)',
    color=C_SEG)

arrow(ax, LX, 7.9 - BH/2, LX, 6.85 + BH/2)

# Step 4: Distance transform
box(ax, LX, 6.85, BW, BH,
    'Distance Transform EDT',
    'dist_smooth: Gaussian σ=3.0 px',
    color=C_SEG)

arrow(ax, LX, 6.85 - BH/2, LX, 5.8 + BH/2)

# Step 5: Peak detection
box(ax, LX, 5.8, BW, BH,
    'Peak Local Max → Markers',
    'min_distance = max(10, 1.2 × L_min) = 10 px',
    color=C_SEG)

arrow(ax, LX, 5.8 - BH/2, LX, 4.75 + BH/2)

# Step 6: Watershed
box(ax, LX, 4.75, BW, BH,
    'Watershed Segmentation',
    'watershed(−dist, markers, mask=binary)',
    color=C_SEG)

arrow(ax, LX, 4.75 - BH/2, LX, 3.7 + BH/2)

# Step 7: Size filter
box(ax, LX, 3.7, BW, BH,
    'Size Filter',
    'MIN_AREA = 86 px²  (½ × L_min × d_mito)',
    color=C_SEG)

arrow(ax, LX, 3.7 - BH/2, LX, 2.65 + BH/2)

# Step 8: Instance label maps
box(ax, LX, 2.65, BW, BH,
    'Instance Label Maps',
    'uint16 TIF · 34 images · 6 models compared',
    color=C_ANAL)

arrow(ax, LX, 2.65 - BH/2, LX, 1.6 + BH/2)

# Step 9: Morphology metrics
box(ax, LX, 1.6, BW, BH,
    'Morphology Quantification (6 metrics)',
    'Area · AR · Eccentricity · Solidity · Tortuosity · Thickness',
    color=C_ANAL)

arrow(ax, LX, 1.6 - BH/2, LX, 0.72 + BH/2 - 0.1)

# Step 10: Statistical analysis
box(ax, LX, 0.72, BW, 0.72,
    'Kruskal-Wallis + Dunn BH-corrected',
    'H=145–187, p<1e-29 (all metrics)',
    color=C_VAL)

# ── TRACK B: Simulation (right column) ───────────────────────────────────────
RX = 13.65   # right column centre x

# Step 1: 3D geometry
box(ax, RX, 10.0, BW, BH,
    '3D Mitochondria Geometry',
    'dot (25%) · rod (45%) · network (30%)\nd_mito=250 nm=2.08 px',
    color=C_SIM)

arrow(ax, RX, 10.0 - BH/2, RX, 8.95 + BH/2)

# Step 2: Fluorophore placement
box(ax, RX, 8.95, BW, BH,
    'Fluorophore Placement',
    'Bernoulli sampling · density=0.6',
    color=C_SIM)

arrow(ax, RX, 8.95 - BH/2, RX, 7.9 + BH/2)

# Step 3: PSF convolution
box(ax, RX, 7.9, BW, BH,
    'PSF Convolution',
    'σ_PSF = 0.61λ/NA/2.355 = 0.876 px\n(λ=488 nm, NA=1.2)',
    color=C_SIM)

arrow(ax, RX, 7.9 - BH/2, RX, 6.85 + BH/2)

# Step 4: Noise model
box(ax, RX, 6.85, BW, BH,
    'Microscope Noise Model',
    'I = Poisson(S·PSF) + N(0, σ_r²)\nS=80 ph/AU · σ_r=8 ADU',
    color=C_SIM)

arrow(ax, RX, 6.85 - BH/2, RX, 5.8 + BH/2)

# Step 5: Physics GT
box(ax, RX, 5.8, BW, BH,
    'Physics-Based Ground Truth',
    'GT = binarise(emitter_mask)\nPSF/noise independent',
    color=C_SIM)

arrow(ax, RX, 5.8 - BH/2, RX, 4.75 + BH/2)

# Step 6: Apply segmentation to synthetic
box(ax, RX, 4.75, BW, BH,
    'Apply Segmentation Pipeline',
    'Same pipeline as Track A\n(N=100 synthetic tiles)',
    color=C_SEG, text_color='white')

arrow(ax, RX, 4.75 - BH/2, RX, 3.7 + BH/2)

# Step 7: Metrics
box(ax, RX, 3.7, BW, BH,
    'Pixel-Level Evaluation',
    'Dice · AJI · F1@IoU=0.5',
    color=C_ANAL)

arrow(ax, RX, 3.7 - BH/2, RX, 2.65 + BH/2)

# Step 8: Validation result
box(ax, RX, 2.65, BW, BH,
    'Synthetic Validation Results',
    'Dice=0.847±0.041\nAJI=0.650±0.106 · F1=0.740±0.149',
    color=C_VAL)

# ── Cross-track arrow: validation confirms pipeline ───────────────────────────
# From right track validation result → left track instance label maps
ax.annotate('', xy=(LX + BW/2 + 0.05, 2.65),
            xytext=(RX - BW/2 - 0.05, 2.65),
            arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2.0,
                            connectionstyle='arc3,rad=0'),
            zorder=5)
ax.text(9.0, 2.85, 'Validates\npipeline', ha='center', va='center',
        fontsize=7.5, color='#C62828', fontweight='bold', zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9.0, 11.35,
        'PhysGT-CLSM: Physics-Informed Annotation-Free GT Generation for CLSM Mitochondria Segmentation',
        ha='center', va='center', fontsize=11.5, fontweight='bold', color='#212121',
        zorder=6)
ax.text(9.0, 11.05,
        'Adapted from Sekh et al., Nature Machine Intelligence 2021  ·  CLSM extensions: PSF calibration, distance-transform watershed, annotation-free inference',
        ha='center', va='center', fontsize=7.8, color='#616161', style='italic', zorder=6)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  label='Input Data'),
    mpatches.Patch(facecolor=C_PREP,   label='Preprocessing'),
    mpatches.Patch(facecolor=C_SIM,    label='Physics Simulation'),
    mpatches.Patch(facecolor=C_SEG,    label='Segmentation'),
    mpatches.Patch(facecolor=C_ANAL,   label='Analysis'),
    mpatches.Patch(facecolor=C_VAL,    label='Output / Validation'),
]
ax.legend(handles=legend_items, loc='lower center',
          bbox_to_anchor=(0.5, -0.04), ncol=6,
          fontsize=8, frameon=True, framealpha=0.9,
          edgecolor='#BDBDBD')

plt.tight_layout(rect=[0, 0.02, 1, 1])

for ext in ('svg', 'pdf'):
    fig.savefig(OUT / f'fig_pipeline.{ext}', bbox_inches='tight',
                dpi=300, format=ext)
    print(f'Saved: {OUT / f"fig_pipeline.{ext}"}')

plt.close(fig)
print('Done.')
