# -*- coding: utf-8 -*-
"""SI figure: residuals of the forward model at the posterior median for all validated cycles.

For every STEP-th cycle (leave-one-day-out estimates of qw and N, nominal nuisance parameters, as in
paper_fig_cycle.py), the measured Tr minus the simulated Tr is computed on a common time axis after
the onset t_s. (a) 5-95 % range of the residual against t - t_s and its median in three ranges of N
(N_BINS), with the inference window shaded. (b) rms residual within the window against N.

Usage: python paper_si_residuals.py  -> Paper/JAerosolSci/figures/fig_s_residuals.{pdf,png}
"""
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from cloudchamber.sbi_shape2 import SHAPE_OFFSETS
from paper_fig_cycle import measured, model
from paper_fig_n_validation import DAYS, load
from paper_style import INK, INK2, tidy

STEP = 5
TAU = np.arange(-0.1, 2.0 + 1e-9, 0.01)  # [s] after t_s
N_BINS = ((300, 1000), (1000, 3000), (3000, 10000))  # [cm-3]


def one(rec):
    row = pd.Series(rec)
    try:
        t, _, tr, p_ini, T_ini, info = measured(row)
        s = model(row, p_ini, T_ini, info)
    except Exception:
        return None
    t_s = row.t_s_shape
    tm = t_s + TAU
    ok = tm <= t[-1]
    r = np.full(TAU.size, np.nan)
    r[ok] = np.interp(tm[ok], t, tr) - np.interp(tm[ok], s["t"], s["Tr"])
    return r


def main():
    d = pd.concat([load(f, a, b) for _, f, a, b, _, _ in DAYS]).iloc[::STEP]
    rows = d.drop(columns=["t"]).to_dict("records")
    with Pool() as pool:
        res = pool.map(one, rows)
    keep = [i for i, r in enumerate(res) if r is not None]
    R = np.array([res[i] for i in keep])
    n = d.nd_cm3.values[keep]
    win = (TAU >= SHAPE_OFFSETS[0] - 1e-9) & (TAU <= SHAPE_OFFSETS[-1] + 1e-9)
    rms = np.sqrt(np.nanmean(R[:, win] ** 2, axis=1))
    late = (TAU >= 1.2) & (TAU <= 1.5)
    print(f"{len(R)} cycles; rms in window median {np.median(rms):.4f} (5-95 % {np.percentile(rms, 5):.4f}-"
          f"{np.percentile(rms, 95):.4f})")
    print(f"median residual in window {np.nanmedian(R[:, win]):+.4f}; at t_s+1.2..1.5 s {np.nanmedian(R[:, late]):+.4f} "
          f"(5-95 % {np.nanpercentile(np.nanmean(R[:, late], 1), 5):+.4f} to {np.nanpercentile(np.nanmean(R[:, late], 1), 95):+.4f})")
    for lo, hi in N_BINS:
        m = (n >= lo) & (n < hi)
        print(f"  N {lo}-{hi}: n {m.sum()}, rms {np.median(rms[m]):.4f}, late {np.nanmedian(np.nanmean(R[m][:, late], 1)):+.4f}")

    ps.apply()
    fig, axs = plt.subplots(1, 2, figsize=(ps.ONE_HALF, 62 * ps.MM))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.93, wspace=0.35)
    ax = axs[0]
    ax.axvspan(SHAPE_OFFSETS[0], SHAPE_OFFSETS[-1], color=ps.GRID, lw=0)
    ax.fill_between(TAU, np.nanpercentile(R, 5, 0), np.nanpercentile(R, 95, 0), color=ps.DAY_COLORS[0], alpha=0.25,
                    lw=0, label="5–95 %")
    for (lo, hi), c in zip(N_BINS, plt.cm.Blues([0.45, 0.7, 0.95])):
        m = (n >= lo) & (n < hi)
        ax.plot(TAU, np.nanmedian(R[m], 0), color=c, lw=1.1,
                label=f"Median, {lo}–{hi}" + (r" cm$^{-3}$" if hi == N_BINS[-1][1] else ""))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel(r"$\mathit{t} - \mathit{t}_\mathrm{s}$ (s)")
    ax.set_ylabel("Measured – simulated Tr")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ps.panel(ax, "a", x=-0.3)
    tidy(ax)
    ax = axs[1]
    ax.scatter(n, rms * 1e3, s=4, color=INK2, lw=0, alpha=0.6, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathit{N}$ (cm$^{-3}$)")
    ax.set_ylabel(r"rms residual in the window ($10^{-3}$)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ps.panel(ax, "b", x=-0.3)
    tidy(ax)
    ps.save(fig, "fig_s_residuals")


if __name__ == "__main__":
    main()
