# -*- coding: utf-8 -*-
"""Paper figure: joint posterior of (qw, N) for four recorded cycles (paper Sect. 4.3).

Each cycle is analysed by the leave-one-day-out model of its day (as in the validation). The four
cycles are arranged 2 x 2 at the text width (140 mm). For each cycle, upper panel: the
measured transmittance Tr = I/I0 after the onset t0 of the pressure drop with the inference window
t_s - 0.1 ... t_s + 0.9 s; lower panel: N_SAMPLES draws of the neural spline flow q(qw, log10 N | y),
shown as a kernel density with the contours enclosing 50 % and 90 % of the posterior mass, the
posterior median and the hygrometer and CPC values of the cycle (dashed lines; error bars: hygrometer
+-0.2 K in dew point, i.e. +-HYG_REL in qw, and CPC +-CPC_REL). N per kg of dry air is converted to
cm-3 at p_ini and T_ini as in the validation. The cycles are chosen by the CPC value only (closest
to TARGETS on the given day), not by their agreement with the references.

Usage: python paper_fig_posterior.py  -> Paper/JAerosolSci/figures/fig_posterior.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

import paper_style as ps
from cloudchamber import conversions as cv
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.sbi_shape2 import SHAPE_OFFSETS
from cloudchamber.shape_estimator import cycle_x
from fit_pressure_correction import raw_path
from paper_fig_n_validation import DAYS, load
from paper_style import HERE, INK, INK2, tidy

# leave-one-day-out model of each day (same order as DAYS)
MODELS = ("npe_shape4_nopcorr-lodo1031_30000.pt", "npe_shape4_nopcorr-lodo1101_30000.pt",
          "npe_shape4_nopcorr-cal1031-1101_30000.pt")
TARGETS = ((0, 500.0), (1, 1500.0), (2, 3000.0), (2, 9000.0))  # (day index, CPC value [cm-3])
N_SAMPLES = 20000
HYG_REL = 2.5e6 * 0.2 / (461.5 * 283.0 ** 2)  # relative qw error of +-0.2 K in dew point (Td ~ 10 C)
CPC_REL = 0.05  # two CPCs agree within 5 %
T_MAX = 3.0  # [s] after t0


def pick(data, day, target):
    d = data[day]
    return d.loc[(np.abs(np.log(d.cpc / target))).idxmin()]


def cycle(row):
    d = np.load(raw_path(row.datetime))
    t, p, tr = d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float)
    p_ini, T_ini = float(d["p_ini"]) * 1e2, float(d["T_ini"])
    x, info = cycle_x(t, p, tr, p_ini, T_ini, offsets=SHAPE_OFFSETS, observed_x=ss4.observed_x)
    return t - info["t0"], tr, x, info, p_ini, T_ini


LIGHT, DARK = 0.18, 0.5  # fraction of the colour mixed into white for the 90 % band and the 50 % region


def tint(color, frac):
    """Opaque mix of `color` with white (frac = 1: the colour itself)."""
    rgb = np.array(to_rgb(color))
    return tuple(1 - frac * (1 - rgb))


def hpd_levels(z, fracs):
    """Density levels whose upper sets contain the fractions `fracs` of the samples."""
    zs = np.sort(z)[::-1]
    return [zs[int(f * (len(zs) - 1))] for f in fracs]


def main():
    data = [load(f, a, b) for _, f, a, b, _, _ in DAYS]
    posts = [torch.load(os.path.join(HERE, "models", m), weights_only=False) for m in MODELS]
    ps.apply()
    fig = plt.figure(figsize=(ps.ONE_HALF, 140 * ps.MM))  # fits one page with the caption
    outer = fig.add_gridspec(2, 2, left=0.12, right=0.98, bottom=0.15, top=0.955, hspace=0.36, wspace=0.32)
    axs = np.empty((2, len(TARGETS)), dtype=object)  # row 0: Tr, row 1: posterior
    for j in range(len(TARGETS)):
        inner = outer[j // 2, j % 2].subgridspec(2, 1, height_ratios=[1, 1.9], hspace=0.6)
        axs[0, j] = fig.add_subplot(inner[0])
        axs[1, j] = fig.add_subplot(inner[1])
    torch.manual_seed(0)
    for j, (day, target) in enumerate(TARGETS):
        row = pick(data, day, target)
        t, tr, x, info, p_ini, T_ini = cycle(row)
        col = ps.DAY_COLORS[day]
        # (top) measured transmittance and inference window
        ax = axs[0, j]
        m = (t >= 0) & (t <= T_MAX)
        t_s = info["t_s"]
        ax.axvspan(t_s + SHAPE_OFFSETS[0], t_s + SHAPE_OFFSETS[-1], color=ps.GRID, lw=0)
        ax.plot(t[m], tr[m], color=INK, lw=0.6)
        ax.set_xlim(0, T_MAX)
        ax.set_xlabel(r"$\mathit{t} - \mathit{t}_0$ (s)")
        if j % 2 == 0:
            ax.set_ylabel("Tr")
        ax.set_title(f"{DAYS[day][0]} {row.datetime[11:16]}", fontsize=8)
        ps.panel(ax, "aceg"[j], x=-0.25, y=1.06)
        tidy(ax)
        # (bottom) joint posterior
        s = posts[day].sample((N_SAMPLES,), x=torch.tensor(x, dtype=torch.float32),
                              show_progress_bars=False).numpy()
        qw, lgn = s[:, 0], s[:, 1] + np.log10(cv.number_per_cm3(1.0, p_ini, T_ini))
        kde = gaussian_kde(np.vstack([qw, lgn]))
        qlo, qhi = np.percentile(qw, [0.5, 99.5])
        nlo, nhi = np.percentile(lgn, [0.5, 99.5])
        pq, pn = 0.15 * (qhi - qlo), 0.15 * (nhi - nlo)
        gq, gn = np.meshgrid(np.linspace(qlo - pq, qhi + pq, 120), np.linspace(nlo - pn, nhi + pn, 120))
        z = kde(np.vstack([gq.ravel(), gn.ravel()])).reshape(gq.shape)
        lev = hpd_levels(kde(np.vstack([qw[:4000], lgn[:4000]])), (0.9, 0.5))
        ax = axs[1, j]
        # opaque tints: 50-90 % band light, inner 50 % region dark (same tints in the legend)
        ax.contourf(gq, 10 ** gn, z, levels=[lev[0], lev[1], z.max()], colors=[tint(col, LIGHT), tint(col, DARK)])
        ax.contour(gq, 10 ** gn, z, levels=lev, colors=[col], linewidths=[0.6, 1.0])
        ax.plot(np.median(qw), 10 ** np.median(lgn), "+", color=col, ms=8, mew=1.2, label="Posterior median")
        qr, nr = row.qw_ref_td_gkg, row.cpc
        ax.axvline(qr, color=INK, lw=0.6, ls="--")
        ax.axhline(nr, color=INK, lw=0.6, ls="--")
        ax.errorbar(qr, nr, xerr=HYG_REL * qr, yerr=CPC_REL * nr, fmt="*", color=INK, ms=7, mec="none",
                    elinewidth=0.8, capsize=2, label="Hygrometer, CPC")
        ax.set_ylim(min(10 ** (nlo - pn), nr * (1 - 1.5 * CPC_REL)), max(10 ** (nhi + pn), nr * (1 + 1.5 * CPC_REL)))
        ax.set_xlim(min(qlo - pq, qr * (1 - 1.5 * HYG_REL)), max(qhi + pq, qr * (1 + 1.5 * HYG_REL)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_xlabel(r"$\mathit{q}_\mathit{w}$ (g kg$^{-1}$)")
        if j % 2 == 0:
            ax.set_ylabel(r"$\mathit{N}$ (cm$^{-3}$)")
        r = np.corrcoef(qw, lgn)[0, 1]
        ax.text(0.97, 0.04, f"$\\mathit{{r}}$ = {r:+.2f}".replace("-", "\u2212"), transform=ax.transAxes, ha="right",
                fontsize=7, color=INK2)
        ps.panel(ax, "bdfh"[j], x=-0.25, y=1.02)
        tidy(ax)
        print(f"{row.datetime}: CPC {row.cpc:.0f}, N med {10**np.median(lgn):.0f} "
              f"(90 % {10**np.percentile(lgn, 5):.0f}-{10**np.percentile(lgn, 95):.0f}); hyg {row.qw_ref_td_gkg:.2f}, "
              f"qw med {np.median(qw):.2f} ({np.percentile(qw, 5):.2f}-{np.percentile(qw, 95):.2f}); "
              f"r(qw, log N) {r:+.2f}; t_s {t_s:.2f} s")
    handles = [Patch(facecolor=tint(INK2, DARK), edgecolor=INK2, lw=1.0, label="50 % of the posterior"),
               Patch(facecolor=tint(INK2, LIGHT), edgecolor=INK2, lw=0.6, label="90 % of the posterior"),
               Line2D([], [], ls="none", marker="+", ms=8, mew=1.2, color=INK2, label="Posterior median"),
               axs[1, 0].get_legend_handles_labels()[0][-1]]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles[:3]] + ["Hygrometer and CPC"],
               loc="lower center", ncol=2, frameon=False, fontsize=7)
    ps.save(fig, "fig_posterior")


if __name__ == "__main__":
    main()
