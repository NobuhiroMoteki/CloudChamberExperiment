# -*- coding: utf-8 -*-
"""SI figure: the parameters of the pressure-curve fit are degenerate, the curve is not.

(a) p_offset against tau_vac2 of the fits to all recorded cycles (31 Oct 12:18 - 2 Nov 16:49), coloured
by frac1; (b) the fitted curves as the relative pressure drop (p_ini - p) / p_ini over 0-4 s after t0
(random subset) with the PRESS_TIMES of the observation vector; (c) their spread at each PRESS_TIME.

Usage: python paper_si_pressure.py  -> Paper/JAerosolSci/figures/fig_s_pressure.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.features import pressure_ramp2
from paper_style import DATA, INK, INK2, tidy

N_SHOW = 300


def main():
    d = pd.read_csv(os.path.join(DATA, "reanalysis", "reanalysis_nopcorr-cal1031-1101.csv"))
    d = d[d.t0_ramp2.notna() & d.p_offset2.notna()]
    t = np.linspace(0, 4, 401)
    drops = np.array([(r.p_ini - pressure_ramp2(t, r.p_ini, 0.0, r.p_offset2, r.tau_vac1, r.frac1, r.tau_vac2,
                                                 r.t_open)) / r.p_ini for r in d.itertuples()])
    at = np.array([np.interp(ss4.PRESS_TIMES, t, x) for x in drops])
    ps.apply()
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE, 62 * ps.MM))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.2, top=0.92, wspace=0.42)
    ax = axs[0]
    sc = ax.scatter(d.tau_vac2, d.p_offset2, c=d.frac1, s=3, cmap="viridis", lw=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\mathit{f}_1$")
    cb.ax.tick_params(labelsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathit{\tau}_2$ (s)")
    ax.set_ylabel(r"$\mathit{p}_\mathrm{off}$ (hPa)")
    ps.panel(ax, "a", x=-0.3)
    tidy(ax)
    ax = axs[1]
    for i in np.random.default_rng(0).choice(len(drops), min(N_SHOW, len(drops)), replace=False):
        ax.plot(t, drops[i] * 100, color=ps.DAY_COLORS[0], lw=0.3, alpha=0.3, rasterized=True)
    ax.plot(t, np.median(drops, 0) * 100, color=INK, lw=1.0, label="Median")
    for tt in ss4.PRESS_TIMES:
        ax.axvline(tt, color=INK2, lw=0.4, ls=":")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"Elapsed time $\mathit{t} - \mathit{t}_0$ (s)")
    ax.set_ylabel(r"$(\mathit{p}_\mathrm{ini} - \mathit{p})/\mathit{p}_\mathrm{ini}$ (%)")
    ax.legend(loc="lower right", frameon=False)
    ps.panel(ax, "b", x=-0.3)
    tidy(ax)
    ax = axs[2]
    rel = at / np.median(at, 0) - 1
    ax.fill_between(ss4.PRESS_TIMES, np.percentile(rel, 5, 0) * 100, np.percentile(rel, 95, 0) * 100,
                    color=ps.DAY_COLORS[0], alpha=0.25, lw=0, label="5–95 %")
    ax.fill_between(ss4.PRESS_TIMES, np.percentile(rel, 25, 0) * 100, np.percentile(rel, 75, 0) * 100,
                    color=ps.DAY_COLORS[0], alpha=0.5, lw=0, label="IQR")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"Elapsed time $\mathit{t} - \mathit{t}_0$ (s)")
    ax.set_ylabel("Drop relative to median (%)")
    ax.legend(loc="upper right", frameon=False)
    ps.panel(ax, "c", x=-0.3)
    tidy(ax)
    ps.save(fig, "fig_s_pressure")
    print(f"{len(d)} cycles; p_offset {d.p_offset2.min():.0f}-{d.p_offset2.max():.0f} hPa, tau2 "
          f"{d.tau_vac2.min():.1f}-{d.tau_vac2.max():.1f} s; drop 5-95 % at 1 s "
          f"{np.percentile(rel[:, 3], 5)*100:+.1f}/{np.percentile(rel[:, 3], 95)*100:+.1f} %, at 4 s "
          f"{np.percentile(rel[:, -1], 5)*100:+.1f}/{np.percentile(rel[:, -1], 95)*100:+.1f} %")


if __name__ == "__main__":
    main()
