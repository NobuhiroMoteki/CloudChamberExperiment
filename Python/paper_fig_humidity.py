# -*- coding: utf-8 -*-
"""SI figure: N / N_CPC against (a) the hygrometer mixing ratio and (b) N_CPC (the remaining variations).

Same cycles and leave-one-day-out estimates as paper_fig_n_validation.py; points per cycle, with
the median and interquartile range in 0.5 g/kg bins and in bins of LOG_BIN decades of N_CPC
(bins with at least MIN_BIN cycles).

Usage: python paper_fig_humidity.py  -> Paper/JAerosolSci/figures/fig_humidity.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from paper_fig_n_validation import DAYS, load
from paper_style import INK, INK2, SURFACE, tidy

BIN = 0.5  # [g/kg]
MIN_BIN = 20


LOG_BIN = 0.125  # [decade] bins of N_CPC


def binned(ax, x, r, edges, centers):
    for lo, hi, c in zip(edges[:-1], edges[1:], centers):
        g = r[(x >= lo) & (x < hi)]
        if len(g) >= MIN_BIN:
            q1, m, q3 = g.quantile([0.25, 0.5, 0.75])
            ax.errorbar(c, m, yerr=[[m - q1], [q3 - m]], fmt="o", ms=4, color=INK, mfc=SURFACE,
                        mew=1.0, elinewidth=1.0, capsize=0, zorder=5)
            yield lo, hi, len(g), m, q1, q3


def main():
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    ps.apply()
    fig, axs = plt.subplots(1, 2, figsize=(ps.ONE_HALF, 70 * ps.MM), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.93, wspace=0.08)
    a = pd.concat([d for _, d, _, _ in data])
    a["r"] = a.nd_cm3 / a.cpc
    for ax, key in zip(axs, ("qw_ref_td_gkg", "cpc")):
        ax.axhline(1.0, color=INK, lw=0.8)
        for f in (0.9, 1.1):
            ax.axhline(f, color=INK2, lw=0.6, ls="--")
        for lab, d, col, mk in data:
            ax.scatter(d[key], d.nd_cm3 / d.cpc, s=4, marker=mk, facecolor=col, edgecolor="none",
                       alpha=0.5, label=lab, rasterized=True)
        tidy(ax)
    # (a) against the hygrometer mixing ratio, 0.5 g/kg bins
    edges = np.arange(np.floor(a.qw_ref_td_gkg.min() / BIN) * BIN, a.qw_ref_td_gkg.max() + BIN, BIN)
    for lo, hi, n, m, q1, q3 in binned(axs[0], a.qw_ref_td_gkg, a.r, edges, edges[:-1] + BIN / 2):
        print(f"  {lo:.1f}-{hi:.1f} g/kg: n {n:4d} median {m:.3f} IQR {q1:.3f}-{q3:.3f}")
    # (b) against N_CPC, bins of LOG_BIN decades
    le = np.arange(np.floor(np.log10(a.cpc.min()) / LOG_BIN) * LOG_BIN, np.log10(a.cpc.max()) + LOG_BIN, LOG_BIN)
    for lo, hi, n, m, q1, q3 in binned(axs[1], a.cpc, a.r, 10 ** le, 10 ** (le[:-1] + LOG_BIN / 2)):
        print(f"  N_CPC {lo:.0f}-{hi:.0f}: n {n:4d} median {m:.3f} IQR {q1:.3f}-{q3:.3f}")
    for lab, d, _, _ in data:
        lr = np.log(d.nd_cm3 / d.cpc)
        print(f"  {lab}: r(ln ratio, ln N_CPC) {np.corrcoef(lr, np.log(d.cpc))[0, 1]:+.2f}")
    axs[0].errorbar([], [], yerr=[], fmt="o", ms=4, color=INK, mfc=SURFACE, mew=1.0, label="Median, IQR")
    axs[0].set_ylim(0.7, 1.4)
    axs[0].set_xlabel(r"$\mathit{q}_{\mathit{w},\mathrm{hyg}}$ (g kg$^{-1}$)")
    axs[0].set_ylabel(r"$\mathit{N}/\mathit{N}_\mathrm{CPC}$")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$\mathit{N}_\mathrm{CPC}$ (cm$^{-3}$)")
    axs[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axs[0].legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.8, markerscale=1.5)
    ps.panel(axs[0], "a", x=-0.18)
    ps.panel(axs[1], "b", x=-0.06)
    ps.save(fig, "fig_humidity")


if __name__ == "__main__":
    main()
