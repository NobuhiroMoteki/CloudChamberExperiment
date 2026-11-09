# -*- coding: utf-8 -*-
"""Paper figure: extinction efficiency of a water droplet at 670 nm, Qext and the effective Qext
seen by the photodiode (light scattered into the cone it subtends is collected; cloudchamber.mie).

(a) Qext and effective Qext against radius, smoothed over a relative radius spread SIGMA_REL as in the
forward model, (b) their ratio.

Usage: python paper_fig_qext.py  -> Paper/JAerosolSci/figures/fig_qext.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps
from cloudchamber import mie
from paper_style import INK, INK2, tidy

SIGMA_REL = 0.04  # relative radius spread used for display (nominal value of the simulator)
A_PLOT = 6e-6  # [m]


def main():
    q = mie.qext_table(SIGMA_REL, forward=False)
    qf = mie.qext_table(SIGMA_REL, forward=True)
    a = np.arange(q.size) * mie.A_STEP
    m = (a > 0.05e-6) & (a <= A_PLOT)
    a_um = a[m] * 1e6
    ps.apply()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(ps.SINGLE, 85 * ps.MM), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.18))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.95)
    ax.plot(a_um, q[m], color=INK2, lw=1.0, label=r"$\mathit{Q}_\mathrm{ext}$")
    ax.plot(a_um, qf[m], color=ps.DAY_COLORS[0], lw=1.0, label=r"$\mathit{Q}_\mathrm{eff}$")
    ax.set_ylabel("Extinction efficiency")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="lower right", frameon=False)
    ps.panel(ax, "a", x=-0.15)
    tidy(ax)
    axr.axhline(1.0, color=INK, lw=0.6)
    axr.plot(a_um, qf[m] / q[m], color=ps.DAY_COLORS[0], lw=1.0)
    axr.set_ylim(0.7, 1.02)
    axr.set_xlim(0, A_PLOT * 1e6)
    axr.set_xlabel("Droplet radius (µm)")
    axr.set_ylabel(r"$\mathit{Q}_\mathrm{eff}/\mathit{Q}_\mathrm{ext}$")
    ps.panel(axr, "b", x=-0.15)
    tidy(axr)
    ps.save(fig, "fig_qext")
    for r in (1, 2, 3, 4, 5):
        i = int(round(r * 1e-6 / mie.A_STEP))
        print(f"  a = {r} um: Qext {q[i]:.3f}, effective {qf[i]:.3f}, ratio {qf[i]/q[i]:.3f}")


if __name__ == "__main__":
    main()
