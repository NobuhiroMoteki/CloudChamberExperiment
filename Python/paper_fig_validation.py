# -*- coding: utf-8 -*-
"""Paper figure: qw and N from the cloud chamber vs the hygrometer and the CPC on one time axis.

Leave-one-day-out estimates and cycles as in paper_fig_n_validation.py (DAYS, load): neither the CPC
nor the hygrometer of the held-out day enters the estimates.
(a) qw with the 90 % credible band and qw / qw_hyg, (b) qw scatter,
(c) N with the 90 % credible band and N / N_CPC, (d) N scatter.

Usage: python paper_fig_validation.py  -> Paper/JAerosolSci/figures/fig_validation.{pdf,png}
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import paper_style as ps
from paper_fig_n_validation import CPC_LINE, DAYS, load, log_ticks
from paper_style import INK, INK2, SURFACE, tidy

QW_LO, QW_HI = 5.0, 10.0  # [g/kg] scatter axes
N_LO, N_HI = 100, 15000  # [cm-3] scatter axes


def time_series(ax, axr, data, y, lo, hi, ref, ref_label, ylim, ylabel, rlim, rticks, band, rlabel,
                days_in_legend=True):
    """Upper panel: reference (thick gray) under the chamber with its band; lower panel: ratio."""
    for lab, d, col, mk in data:
        seg = (d.t.diff() > pd.Timedelta("5min")).cumsum()  # do not join across gaps
        for k, (_, g) in enumerate(d.groupby(seg)):
            first = k == 0 and lab == "31 Oct"
            ax.plot(g.t, g[ref], color=CPC_LINE, lw=2.2, solid_capstyle="round", label=ref_label if first else None)
            ax.fill_between(g.t, g[lo], g[hi], color=col, alpha=0.3, lw=0)
            ax.plot(g.t, g[y], color=col, lw=0.7,
                    label=f"Chamber, {lab}" if k == 0 and days_in_legend else None)
        axr.scatter(d.t, d[y] / d[ref], s=1.5, color=col, lw=0, rasterized=True)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, ncol=4, handlelength=1.5, columnspacing=1.2)
    plt.setp(ax.get_xticklabels(), visible=False)
    tidy(ax)
    axr.axhline(1.0, color=INK, lw=0.8)
    for f in (1 - band, 1 + band):
        axr.axhline(f, color=INK2, lw=0.6, ls="--")
    axr.set_ylim(*rlim)
    axr.set_yticks(rticks)
    axr.set_ylabel(rlabel)
    tidy(axr)


def scatter(ax, data, x, y, lo, hi, band, text, xlabel, ylabel):
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
    for f in (1 - band, 1 + band):
        ax.plot([lo, hi], [f * lo, f * hi], color=INK2, lw=0.6, ls="--")
    for lab, d, col, mk in data:
        ax.scatter(d[x], d[y], s=8, marker=mk, facecolor=col, edgecolor=SURFACE, lw=0.3, alpha=0.8,
                   label=lab, rasterized=True)
    ax.text(0.04, 0.97, text, transform=ax.transAxes, va="top", color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    tidy(ax)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.2, markerscale=1.3)


def main():
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    a = pd.concat([d for _, d, _, _ in data])
    rq = a.qw_gkg / a.qw_ref_td_gkg
    rn = a.nd_cm3 / a.cpc
    ps.apply()
    fig = plt.figure(figsize=(ps.DOUBLE, 170 * ps.MM))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.1, 1], height_ratios=[2.6, 1, 0.4, 2.6, 1], wspace=0.3,
                          hspace=0.12, left=0.075, right=0.985, bottom=0.075, top=0.965)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_qr = fig.add_subplot(gs[1, 0], sharex=ax_q)
    ax_n = fig.add_subplot(gs[3, 0], sharex=ax_q)
    ax_nr = fig.add_subplot(gs[4, 0], sharex=ax_q)

    time_series(ax_q, ax_qr, data, "qw_gkg", "qw_lo", "qw_hi", "qw_ref_td_gkg", "Hygrometer", (QW_LO, 10.8),
                r"$\mathit{q}_\mathit{w}$ (g kg$^{-1}$)", (0.9, 1.1), [0.95, 1.0, 1.05], 0.05, r"$\mathit{q}_\mathit{w}/\mathit{q}_{\mathit{w},\mathrm{hyg}}$")
    time_series(ax_n, ax_nr, data, "nd_cm3", "nd_cm3_lo", "nd_cm3_hi", "cpc", "CPC", (N_LO, 40000),
                r"$\mathit{N}$ (cm$^{-3}$)", (0.6, 1.4), [0.6, 0.8, 1.0, 1.2, 1.4], 0.1, r"$\mathit{N}/\mathit{N}_\mathrm{CPC}$",
                days_in_legend=False)  # colours of the days as in (a)
    ax_n.set_yscale("log")
    ax_n.set_ylim(N_LO, 40000)  # room for the legend
    log_ticks(ax_n.yaxis)
    plt.setp(ax_qr.get_xticklabels(), visible=False)  # one time axis, at the bottom
    ax_nr.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax_nr.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 18]))
    ax_nr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%-d %b"))
    ps.panel(ax_q, "a", x=-0.075)
    ps.panel(ax_n, "c", x=-0.075)

    ax = fig.add_subplot(gs[0:2, 1])
    scatter(ax, data, "qw_ref_td_gkg", "qw_gkg", QW_LO, QW_HI, 0.05,
            f"$\\mathit{{q}}_\\mathit{{w}}/\\mathit{{q}}_{{\\mathit{{w}},\\mathrm{{hyg}}}}$ = {rq.median():.3f}\n(IQR {rq.quantile(.25):.3f}–{rq.quantile(.75):.3f})\n"
            f"$\\mathit{{r}}$ = {a.qw_gkg.corr(a.qw_ref_td_gkg):.3f}, $\\mathit{{n}}$ = {len(a)}",
            r"Hygrometer, $\mathit{q}_{\mathit{w},\mathrm{hyg}}$ (g kg$^{-1}$)", r"Cloud chamber, $\mathit{q}_\mathit{w}$ (g kg$^{-1}$)")
    ps.panel(ax, "b", x=-0.27)
    ax = fig.add_subplot(gs[3:5, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    scatter(ax, data, "cpc", "nd_cm3", N_LO, N_HI, 0.1,
            f"$\\mathit{{N}}/\\mathit{{N}}_\\mathrm{{CPC}}$ = {rn.median():.3f}\n(IQR {rn.quantile(.25):.3f}–{rn.quantile(.75):.3f})\n"
            f"$\\mathit{{r}}$ = {a.nd_cm3.corr(a.cpc):.3f}, $\\mathit{{n}}$ = {len(a)}", r"CPC, $\mathit{N}_\mathrm{CPC}$ (cm$^{-3}$)", r"Cloud chamber, $\mathit{N}$ (cm$^{-3}$)")
    log_ticks(ax.xaxis)
    log_ticks(ax.yaxis)
    ps.panel(ax, "d", x=-0.27)

    ps.save(fig, "fig_validation")
    print(f"n {len(a)}, qw/hyg {rq.median():.4f} (IQR {rq.quantile(.25):.4f}-{rq.quantile(.75):.4f}), "
          f"r {a.qw_gkg.corr(a.qw_ref_td_gkg):.4f}; N/CPC {rn.median():.3f} "
          f"(IQR {rn.quantile(.25):.3f}-{rn.quantile(.75):.3f}), r {a.nd_cm3.corr(a.cpc):.4f}")
    # variations of N/N_CPC: medians over consecutive 3-h intervals (at least 90 % of the cycles present)
    blocks = rn.groupby(a.t.dt.floor("3h")).agg(["median", "count"])
    blocks = blocks[blocks["count"] >= 0.9 * 180]
    print(f"N/CPC medians over {len(blocks)} 3-h intervals: {blocks['median'].min():.3f}-{blocks['median'].max():.3f}")


if __name__ == "__main__":
    main()
