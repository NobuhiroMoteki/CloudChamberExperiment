# -*- coding: utf-8 -*-
"""Common style of the paper figures (Elsevier artwork rules, Journal of Aerosol Science).

Widths 90 mm (single column), 140 mm (1.5 column), 190 mm (double column). Vector PDF with the
fonts embedded (TrueType, Arial); dense point clouds are rasterised inside the PDF at RASTER_DPI
(>= 500 dpi for combination art). Text 8 pt, ticks 7 pt (nothing smaller). A PNG preview is
written next to the PDF.
"""
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
FIG_DIR = os.path.join(HERE, os.pardir, "Paper", "JAerosolSci", "figures")
MM = 1 / 25.4  # [inch]
SINGLE, ONE_HALF, DOUBLE = 90 * MM, 140 * MM, 190 * MM
RASTER_DPI = 600

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF = "#9a9994"  # reference instrument (CPC, hygrometer)
DAY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, fixed order
DAY_MARKERS = ("o", "s", "^")


def apply():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.0})


def tidy(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, label, x=-0.12, y=1.02):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontweight="bold", va="bottom", fontsize=8)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), dpi=RASTER_DPI)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    print(f"saved {os.path.abspath(os.path.join(FIG_DIR, name))}.pdf/.png")
