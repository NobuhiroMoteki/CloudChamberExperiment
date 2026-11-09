# -*- coding: utf-8 -*-
"""Paper figure: the posterior on held-out simulations (the last N_TEST of the training set, not
used in training) of the model calibrated on 10/31-11/01 (npe_shape4_nopcorr-cal1031-1101_30000.pt).

(a) qw, (b) N: posterior median with the 90 % credible interval against the true value (a random
subset of N_SHOW points drawn), (c) coverage of central credible intervals of every level.

Usage: python paper_fig_heldout.py  -> Paper/JAerosolSci/figures/fig_heldout.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedLocator, FuncFormatter

import paper_style as ps
from cloudchamber import conversions as cv
from cloudchamber import sbi_shape4 as ss4
from paper_style import HERE, INK, INK2, tidy

TAG = "_nopcorr-cal1031-1101"
N_SIMS, N_TEST, N_SAMPLES, N_SHOW = 30000, 2000, 1000, 300
LEVELS = np.linspace(0.05, 0.95, 19)


def main():
    d = np.load(os.path.join(HERE, "models", f"sbi_shape4{TAG}_sims_{N_SIMS}.npz"))
    th, x = d["theta"][-N_TEST:], d["x"][-N_TEST:]
    post = torch.load(os.path.join(HERE, "models", f"npe_shape4{TAG}_{N_SIMS}.pt"), weights_only=False)
    s = post.sample_batched((N_SAMPLES,), x=torch.tensor(x, dtype=torch.float32), show_progress_bars=False).numpy()
    med, lo, hi = np.median(s, 0), np.percentile(s, 5, 0), np.percentile(s, 95, 0)
    # N per kg -> cm-3 at the initial state (as for the measurements)
    k = cv.number_per_cm3(1.0, x[:, ss4.X_NAMES.index("p_ini_hPa")] * 1e2, x[:, ss4.X_NAMES.index("T_ini")])
    rank = (s < th[None]).mean(0)  # position of the true value in the posterior
    cov = np.array([[np.mean(np.abs(rank[:, j] - 0.5) <= lv / 2) for lv in LEVELS] for j in range(2)])
    rms_qw = np.sqrt(np.mean((med[:, 0] - th[:, 0]) ** 2))
    rms_n = np.sqrt(np.mean((med[:, 1] - th[:, 1]) ** 2))
    i90 = int(np.argmin(np.abs(LEVELS - 0.9)))
    print(f"90% coverage qw {cov[0, i90]*100:.1f} %, N {cov[1, i90]*100:.1f} %; rms qw {rms_qw:.3f} g/kg, "
          f"log10 N {rms_n:.4f} ({(10**rms_n-1)*100:.1f} %)")

    ps.apply()
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE, 66 * ps.MM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.92, wspace=0.42)
    show = np.random.default_rng(0).choice(N_TEST, N_SHOW, replace=False)
    col = ps.DAY_COLORS[0]
    # (a) qw
    ax = axs[0]
    t, m = th[show, 0], med[show, 0]
    ax.errorbar(t, m, yerr=[m - lo[show, 0], hi[show, 0] - m], fmt="o", ms=2, color=col, elinewidth=0.4,
                alpha=0.7, rasterized=True)
    ax.plot([3, 20], [3, 20], color=INK, lw=0.8)
    ax.set_xlim(3, 20)
    ax.set_ylim(3, 20)
    ax.set_aspect("equal")
    ax.set_xlabel(r"True $\mathit{q}_\mathit{w}$ (g kg$^{-1}$)")
    ax.set_ylabel(r"Inferred $\mathit{q}_\mathit{w}$ (g kg$^{-1}$)")
    ax.text(0.04, 0.96, f"rms {rms_qw:.2f} g kg$^{{-1}}$", transform=ax.transAxes, va="top")
    ps.panel(ax, "a", x=-0.3)
    tidy(ax)
    # (b) N
    ax = axs[1]
    kk = k[show]
    t, m = 10 ** th[show, 1] * kk, 10 ** med[show, 1] * kk
    ax.errorbar(t, m, yerr=[m - 10 ** lo[show, 1] * kk, 10 ** hi[show, 1] * kk - m], fmt="o", ms=2, color=col,
                elinewidth=0.4, alpha=0.7, rasterized=True)
    ax.plot([50, 2e5], [50, 2e5], color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 1.5e5)
    ax.set_ylim(80, 1.5e5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([100, 1000, 10000, 100000]))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_aspect("equal")
    ax.set_xlabel(r"True $\mathit{N}$ (cm$^{-3}$)")
    ax.set_ylabel(r"Inferred $\mathit{N}$ (cm$^{-3}$)")
    ax.text(0.04, 0.96, f"rms {(10**rms_n-1)*100:.1f} %", transform=ax.transAxes, va="top")
    ps.panel(ax, "b", x=-0.3)
    tidy(ax)
    # (c) coverage
    ax = axs[2]
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8)
    ax.plot(LEVELS, cov[0], "o-", ms=3, color=col, label=r"$\mathit{q}_\mathit{w}$")
    ax.plot(LEVELS, cov[1], "s--", ms=3, color=ps.DAY_COLORS[1], label=r"$\mathit{N}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Credible level")
    ax.set_ylabel("Empirical coverage")
    ax.legend(loc="lower right", frameon=False)
    ps.panel(ax, "c", x=-0.3)
    tidy(ax)
    ps.save(fig, "fig_heldout")


if __name__ == "__main__":
    main()
