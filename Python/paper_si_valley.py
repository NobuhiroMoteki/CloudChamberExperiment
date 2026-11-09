# -*- coding: utf-8 -*-
"""SI figure: the pressure-sensor gain and lag are degenerate along a valley.

Superseded 2026-11-08: the paper no longer uses the pressure-sensor correction (sbi_shape4.PCORR).

From the grids of fit_pressure_correction.py (models/pressure_correction_grid_fwd_ac*.npz, 400 cycles,
qw fixed at the hygrometer, effective Qext): cost per cycle (dt_s / 20 ms)^2 + (rms Tr / 0.003)^2,
minimised over log10 N, summed over cycles. (a) the summed cost against gain and lag (alpha_c = 0.3),
with the valley and the prior of the simulator; (b) the best lag at each gain for three alpha_c.

Usage: python paper_si_valley.py  -> Paper/JAerosolSci/figures/fig_s_valley.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps
from cloudchamber import sbi_shape4 as ss4
from paper_style import HERE, INK, INK2, tidy

ALPHAS = ("0.1", "0.3", "1.0")


def total_cost(ac):
    d = np.load(os.path.join(HERE, "models", f"pressure_correction_grid_fwd_ac{ac}.npz"))
    c = (d["dts"] / 0.02) ** 2 + (d["rms"] / 0.003) ** 2
    c = np.where(np.isfinite(c), c, 1e6).min(axis=2)  # (cycle, gain, lag), best log10 N
    return d["gains"], d["lags"], c.sum(0), len(c)


def valley(g):
    return ss4.P_LAG_REF + ss4.LAG_SLOPE * (g - ss4.P_GAIN_REF)


def main():
    ps.apply()
    fig, axs = plt.subplots(1, 2, figsize=(ps.ONE_HALF, 62 * ps.MM))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.92, wspace=0.55)
    G, L, tot, n = total_cost("0.3")
    ax = axs[0]
    rel = np.log10(tot / tot.min())
    pc = ax.pcolormesh(G, L, rel.T, cmap="Blues_r", vmin=0, vmax=1.5, shading="nearest", rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label("log$_{10}$(cost / min)")
    cb.ax.tick_params(labelsize=7)
    g = np.linspace(G.min(), G.max(), 50)
    ax.plot(g, valley(g), color=ps.DAY_COLORS[1], lw=1.0, label="Valley")
    gp = np.linspace(*ss4.P_GAIN, 20)
    ax.fill_between(gp, valley(gp) - ss4.LAG_HALF, valley(gp) + ss4.LAG_HALF, color=ps.DAY_COLORS[1], alpha=0.3,
                    lw=0, label="Prior")
    ax.plot(1.0, 0.0, "x", color=INK, ms=5, label="No correction")
    ax.set_ylim(-0.1, 0.3)
    ax.set_xlabel("Gain (true / measured drop)")
    ax.set_ylabel("Lag (s)")
    ax.legend(loc="upper right", frameon=False)
    ps.panel(ax, "a", x=-0.28)
    print(f"{n} cycles; best (alpha_c 0.3) gain/lag", G[np.unravel_index(tot.argmin(), tot.shape)[0]],
          L[np.unravel_index(tot.argmin(), tot.shape)[1]])
    ax = axs[1]
    for ac, col, mk in zip(ALPHAS, ps.DAY_COLORS, ps.DAY_MARKERS):
        G, L, tot, _ = total_cost(ac)
        ax.plot(G, L[tot.argmin(1)], marker=mk, ms=3, color=col, lw=0.8, label=rf"$\mathit{{\alpha}}_\mathit{{c}}$ = {ac}")
        print(f"  alpha_c {ac}: lag at gain 0.96 {L[tot[list(np.round(G, 3)).index(0.96)].argmin()]:.2f} s")
    ax.plot(g, valley(g), color=INK2, lw=0.8, ls="--", label="Valley (simulator)")
    ax.set_xlabel("Gain (true / measured drop)")
    ax.set_ylabel("Best lag (s)")
    ax.legend(loc="upper right", frameon=False)
    ps.panel(ax, "b", x=-0.3)
    tidy(ax)
    ps.save(fig, "fig_s_valley")


if __name__ == "__main__":
    main()
