# -*- coding: utf-8 -*-
"""Paper figure: the instrument. (a) photograph, (b) schematic of the air paths, optics and signals.

The photograph (Paper/JAerosolSci/figures/photo_instrument_src.jpg, taken by the author) is re-saved
without its metadata (EXIF) and placed next to a schematic drawn here.

Usage: python paper_fig_instrument.py  -> Paper/JAerosolSci/figures/fig_instrument.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

import paper_style as ps
from paper_style import FIG_DIR, INK, INK2

SRC = os.path.join(FIG_DIR, "photo_instrument_src.jpg")
PHOTO = os.path.join(FIG_DIR, "photo_instrument.jpg")
AIR, LIGHT, SIG = "#2a78d6", "#e34948", INK2
CTRL_LS = ":"  # line style of the valve control (signals are dashed)


def photo():
    """Photograph without metadata, cropped to the apparatus and scaled for 300 dpi at its size."""
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    img = img.crop((int(0.2 * w), int(0.02 * h), w, int(0.95 * h)))
    img.thumbnail((1200, 1600))
    img.save(PHOTO, quality=92)  # no exif argument: metadata dropped
    return np.asarray(img)


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=6.5):
    # no padding, so the drawn border is the given rectangle; drawn above the lines, so a line that
    # ends on the border never shows inside the box
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec,
                                lw=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(ax, xy0, xy1, color, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def line(ax, pts, color, lw=1.2, ls="-"):
    p = np.array(pts)
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round")


def schematic(ax):
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.01, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    cx, cy, r = 0.50, 0.42, 0.20
    xi, xt, xp, xe = cx - 0.045, cx - 0.015, cx + 0.015, cx + 0.045  # inlet, thermistor, pressure, exhaust
    top = cy + r + 0.20  # top of the stopper
    y_air = 0.90  # height of the valve line
    x_v2c, x_pd, x_ps, x_th = 0.75, 0.80, 0.835, 0.865  # vertical signal lines (left of the pump)
    # flask, neck and stopper
    ax.add_patch(Circle((cx, cy), r, fc="#f4f8fd", ec=INK, lw=1.0))
    ax.add_patch(Rectangle((cx - 0.065, cy + r - 0.01), 0.13, 0.16, fc="#f4f8fd", ec=INK, lw=1.0))
    ax.add_patch(Rectangle((cx - 0.075, cy + r + 0.15), 0.15, 0.05, fc="#dcdad3", ec=INK, lw=0.8))
    ax.text(cx, cy - r + 0.04, "Flask", ha="center", fontsize=6.5)
    # room-air inlets of the chamber, the CPC and the hygrometer, side by side at one place
    x_in = 0.0  # tube openings
    y_ch, y_hyg, y_cpc = 0.86, 0.825, 0.79
    ax.text(-0.045, y_ch + 0.03, "Room air", fontsize=6.5, ha="left")
    for y in (y_ch, y_cpc, y_hyg):
        arrow(ax, (-0.045, y), (x_in - 0.008, y), AIR, lw=0.7)
    # chamber inlet path: -> V1 -> long tube into the flask
    line(ax, [(x_in, y_ch), (0.12, y_ch), (0.12, y_air), (0.19, y_air)], AIR)
    box(ax, 0.19, y_air - 0.03, 0.12, 0.06, "Valve 1")
    line(ax, [(0.31, y_air), (xi, y_air), (xi, cy + r - 0.005)], AIR)  # inlet ends near the end of the neck
    # flow direction, once at each tube opening inside the flask (thin, clear of the tubes)
    arrow(ax, (xi, cy + r - 0.015), (xi, cy + r - 0.07), AIR, lw=0.7)
    arrow(ax, (xe, cy + r + 0.005), (xe, cy + r + 0.07), AIR, lw=0.7)
    # exhaust path: short tube -> V2 -> vacuum pump -> exhaust
    line(ax, [(xe, cy + r + 0.085), (xe, y_air), (0.66, y_air)], AIR)  # outlet ends higher in the neck
    box(ax, 0.66, y_air - 0.03, 0.12, 0.06, "Valve 2")
    line(ax, [(0.78, y_air), (0.90, y_air)], AIR)
    box(ax, 0.90, y_air - 0.045, 0.19, 0.09, "Vacuum\npump", fs=6.5)
    line(ax, [(0.995, y_air - 0.045), (0.995, 0.74)], AIR)
    arrow(ax, (0.995, 0.732), (0.995, 0.675), AIR, lw=0.7)  # past the open end, clear of the tube
    ax.text(1.005, 0.70, "Exhaust", fontsize=6, ha="left", va="center")
    # pressure port (stub through the stopper, tubing to the sensor)
    line(ax, [(xp, cy + r + 0.13), (xp, 1.00), (0.58, 1.00)], INK2, lw=0.9)
    box(ax, 0.58, 0.975, 0.205, 0.05, "Pressure sensor", fs=6)
    line(ax, [(0.785, 1.00), (x_ps, 1.00), (x_ps, 0.22)], SIG, lw=0.8, ls="--")
    # thermistor: thin-film element on the tip of its own tube; wires leave at the top
    line(ax, [(xt, cy + 0.08), (xt, top + 0.02)], "#9a9994", lw=1.4)
    ax.add_patch(Rectangle((xt - 0.006, cy + 0.035), 0.012, 0.045, fc="#d9a441", ec=INK, lw=0.5))
    ax.text(xt + 0.02, cy + 0.055, "Thermistor", fontsize=6, va="center")
    line(ax, [(xt, top + 0.02), (xt, 1.05), (x_th, 1.05), (x_th, 0.22)], SIG, lw=0.8, ls="--")
    # optics: laser -> centre -> photodiode
    box(ax, 0.03, cy - 0.03, 0.17, 0.06, "Laser", fs=6.5)
    line(ax, [(0.20, cy), (cx + r, cy)], LIGHT, lw=1.0)
    arrow(ax, (0.33, cy), (0.39, cy), LIGHT, lw=1.0)
    ax.add_patch(Rectangle((cx + r, cy - 0.025), 0.018, 0.05, fc=INK, ec=INK))
    ax.text(cx + r - 0.012, cy - 0.025, "Photodiode", fontsize=6, va="top", ha="right")
    line(ax, [(cx + r + 0.018, cy), (x_pd, cy), (x_pd, 0.22)], SIG, lw=0.8, ls="--")
    # acquisition and control
    box(ax, 0.73, 0.15, 0.27, 0.07, "DAQ device", fs=6.5)
    box(ax, 0.73, 0.01, 0.27, 0.08, "Computer", fs=6.5)
    line(ax, [(0.865, 0.15), (0.865, 0.09)], SIG, lw=0.8, ls="--")
    ax.text(0.875, 0.12, "USB", fontsize=6, color=INK2, va="center")
    ax.text(x_th + 0.01, 0.27, "Signals", fontsize=6, color=INK2)
    line(ax, [(0.73, 0.185), (0.265, 0.185), (0.265, y_air - 0.03)], SIG, lw=0.9, ls=CTRL_LS)
    line(ax, [(x_v2c, 0.22), (x_v2c, y_air - 0.03)], SIG, lw=0.9, ls=CTRL_LS)
    ax.text(0.30, 0.15, "Valve control", fontsize=6, color=INK2)
    # reference instruments side by side, each with its own inlet tube (one bend, entering from the top)
    y_box = 0.62
    for y_open, x0, w, name in ((y_cpc, 0.0, 0.08, "CPC"), (y_hyg, 0.095, 0.135, "Hygrometer")):
        box(ax, x0, y_box, w, 0.07, name, fc="#f3f2ee", fs=5.8)
        xc = x0 + w / 2
        line(ax, [(x_in, y_open), (xc, y_open), (xc, y_box + 0.07)], AIR)
    ax.legend(handles=[plt.Line2D([], [], color=AIR, lw=1.2, label="Air"),
                       plt.Line2D([], [], color=SIG, lw=0.8, ls="--", label="Signals"),
                       plt.Line2D([], [], color=SIG, lw=0.9, ls=CTRL_LS, label="Control")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=6, handlelength=1.6)


def main():
    ps.apply()
    img = photo()
    fig = plt.figure(figsize=(ps.DOUBLE, 100 * ps.MM))
    ax_p = fig.add_axes([0.0, 0.02, 0.36, 0.93])
    ax_p.imshow(img)
    ax_p.axis("off")
    ps.panel(ax_p, "a", x=0.0, y=1.0)
    ax_s = fig.add_axes([0.38, 0.0, 0.62, 0.97])
    schematic(ax_s)
    ps.panel(ax_s, "b", x=0.02, y=0.98)
    ps.save(fig, "fig_instrument")


if __name__ == "__main__":
    main()
