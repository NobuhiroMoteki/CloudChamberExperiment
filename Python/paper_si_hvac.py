# -*- coding: utf-8 -*-
"""SI figure: room-air experiments of 2 Nov (window, air conditioning, air purifier).

Cycles of the N validation (held-out model for 2 Nov) from 09:30 on (laser realigned at 09:27).
(a) N and N_CPC, (b) N / N_CPC per cycle; shaded: air conditioning off; bars on top: window open,
air purifier on. States from the experiment log.

Usage: python paper_si_hvac.py  -> Paper/JAerosolSci/figures/fig_s_hvac.{pdf,png}
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import paper_style as ps
from paper_fig_n_validation import DAYS, load
from paper_style import INK, INK2, tidy

DAY = "2026-11-02"
START = "09:30"
WINDOW_OPEN = [("09:24", "10:02"), ("10:17", "12:02"), ("14:40", "15:45")]
AC_OFF = [("10:46", "12:45"), ("14:35", "15:45")]
PURIFIER_ON = [("09:24", "10:21"), ("13:42", "14:17"), ("15:45", "16:50")]
MEDIANS = [("09:30", "10:46", "AC on"), ("10:46", "12:45", "AC off"), ("12:55", "14:17", "AC on"),
           ("14:40", "15:45", "AC off"), ("15:55", "16:50", "AC on")]


def ts(hm):
    return pd.Timestamp(f"{DAY} {hm}")


def main():
    lab, f, a, b, col, _ = DAYS[2]
    d = load(f, a, b)
    d = d[d.t >= ts(START)]
    d["r"] = d.nd_cm3 / d.cpc
    ps.apply()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(ps.ONE_HALF, 95 * ps.MM), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.1))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.88)
    for x in (ax, axr):
        for s, e in AC_OFF:
            x.axvspan(ts(s), ts(e), color=ps.GRID, lw=0)
    seg = (d.t.diff() > pd.Timedelta("5min")).cumsum()
    for k, (_, g) in enumerate(d.groupby(seg)):
        ax.plot(g.t, g.cpc, color=ps.REF, lw=2.2, solid_capstyle="round", label="CPC" if k == 0 else None)
        ax.plot(g.t, g.nd_cm3, color=col, lw=0.8, label="Chamber" if k == 0 else None)
    axr.scatter(d.t, d.r, s=2, color=col, lw=0, rasterized=True)
    for s, e, name in MEDIANS:
        g = d[(d.t >= ts(s)) & (d.t < ts(e))]
        axr.plot([ts(s), ts(e)], [g.r.median()] * 2, color=INK, lw=1.2)
        axr.text(ts(s) + (ts(e) - ts(s)) / 2, 1.22, f"{g.r.median():.2f}", ha="center", fontsize=7)
        print(f"{s}-{e} {name}: n {len(g)}, N/CPC {g.r.median():.3f}, CPC {g.cpc.median():.0f}")
    # state bars above the top panel
    for row, (spans, name) in enumerate([(WINDOW_OPEN, "Window open"), (PURIFIER_ON, "Purifier on")]):
        y = 1.04 + 0.06 * row
        for s, e in spans:
            ax.plot([ts(s), ts(e)], [y, y], transform=ax.get_xaxis_transform(), color=INK2, lw=2.5,
                    solid_capstyle="butt", clip_on=False)
        ax.text(-0.01, y, name, transform=ax.transAxes, fontsize=7, va="center", ha="right")
    ax.set_ylabel(r"$\mathit{N}$ (cm$^{-3}$)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False)
    ps.panel(ax, "a", x=-0.14, y=1.16)
    tidy(ax)
    axr.axhline(1.0, color=INK, lw=0.6)
    for v in (0.9, 1.1):
        axr.axhline(v, color=INK2, lw=0.6, ls="--")
    axr.set_ylim(0.75, 1.3)
    axr.set_ylabel(r"$\mathit{N}/\mathit{N}_\mathrm{CPC}$")
    axr.set_xlabel("Time on 2 Nov (shaded: air conditioning off)")
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ps.panel(axr, "b", x=-0.14)
    tidy(axr)
    ps.save(fig, "fig_s_hvac")


if __name__ == "__main__":
    main()
