# -*- coding: utf-8 -*-
"""Paper figure (Sect. 2.4): pressure during the expansion and the fit of Eq. (1).

(a) Measured pressure (3-point running median) of the high-N cycle of fig_cycle with the fit of Eq. (1)
and of a single exponential with instantaneous opening (paper_pressure_models.py, model A).
(b) Their residuals. (c) rms residual over the first 2 s after t0 of the four models of
paper_pressure_models.py for all cycles of the evaluation (read from pressure_models_rms.npz).
Time is from the onset t0 of the fit of Eq. (1).

Usage: python paper_fig_pressure.py  -> Paper/JAerosolSci/figures/fig_pressure.{pdf,png}
(run paper_pressure_models.py first)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from cloudchamber.features import despike, find_decompression_start, pressure_ramp2
from paper_fig_cycle import measured, pick
from paper_fig_n_validation import DAYS, load
from paper_pressure_models import DP, MODELS, RMS_FILE
from paper_pressure_models import fit as fit_pressure_model
from paper_style import INK, INK2, tidy

T_MAX = 10.0  # [s] the whole expansion phase
ZOOM = 0.6  # [s] end of the inset (the valve opening)
DECIMATE = 20  # measured samples shown in (a): 5 Hz of the 100 Hz record
LABELS = {"A": "Single exponential,\ninstantaneous opening", "B": "Single exponential,\nvalve opening",
          "C": "Two exponentials,\ninstantaneous opening", "D": "Eq. (1)"}


def main():
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    row = pick(data)[1]  # the high-N cycle of fig_cycle
    t, p, _, p_ini, _, info = measured(row)  # t from t0 of the fit of Eq. (1)
    pd_ = despike(p)  # shape: (n_samples,) [Pa]
    p_fit = pressure_ramp2(t, p_ini, 0.0, info["p_offset"], info["tau_vac"], info["frac1"],
                           info["tau_vac2"], info["t_open"])
    par_a, _, _ = fit_pressure_model(t, pd_, p_ini, find_decompression_start(pd_, p_ini, DP), MODELS["A"])
    p_a = pressure_ramp2(t, p_ini, *par_a)
    rms = np.load(RMS_FILE)["rms"][:, :, 0]  # shape: (n_cycles, 4) [hPa], window 0-2 s

    ps.apply()
    fig = plt.figure(figsize=(ps.DOUBLE, 95 * ps.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1.7, 1], wspace=0.45, hspace=0.12,
                          left=0.08, right=0.98, bottom=0.12, top=0.91)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_p)
    m = (t >= 0) & (t <= T_MAX)
    k = np.nonzero(m)[0][::DECIMATE]  # every DECIMATE-th sample, so that the curves stay visible
    ax_p.plot(t[k], pd_[k] * 1e-2, "o", ms=3.2, mfc="white", mec=INK2, mew=0.6, zorder=2,
              label=f"Measured (every {DECIMATE}th sample)")
    ax_p.plot(t[m], p_fit[m] * 1e-2, color=ps.DAY_COLORS[0], lw=1.0, zorder=3, label="Eq. (1)")
    ax_p.plot(t[m], p_a[m] * 1e-2, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4, label="Single exponential")
    ax_p.set_ylabel("Pressure (hPa)")
    # inset: the valve opening, all samples
    zi = (t >= 0) & (t <= ZOOM)
    axin = ax_p.inset_axes([0.47, 0.33, 0.5, 0.36])
    axin.set_facecolor("white")
    axin.set_zorder(6)
    axin.plot(t[zi], pd_[zi] * 1e-2, "o", ms=2.2, mfc="white", mec=INK2, mew=0.5, zorder=2)
    axin.plot(t[zi], p_fit[zi] * 1e-2, color=ps.DAY_COLORS[0], lw=1.0, zorder=3)
    axin.plot(t[zi], p_a[zi] * 1e-2, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    axin.set_xlim(0, ZOOM)
    axin.tick_params(labelsize=6, length=2, pad=1)
    tidy(axin)
    lo, hi = pd_[zi].min() * 1e-2, pd_[zi].max() * 1e-2
    ax_p.add_patch(plt.Rectangle((0, lo), ZOOM, hi - lo, fill=False, ec=INK2, lw=0.6, zorder=5))
    ax_p.legend(loc="upper right", frameon=False)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    ax_r.axhline(0, color=INK2, lw=0.5)
    ax_r.plot(t[m], (pd_[m] - p_a[m]) * 1e-2, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax_r.plot(t[m], (pd_[m] - p_fit[m]) * 1e-2, color=ps.DAY_COLORS[0], lw=0.8)
    ax_r.set_ylim(-12, 12)
    ax_r.set_xlim(0, min(T_MAX, t[np.isfinite(pd_)].max()))  # the record ends 10 s after the valve command
    ax_r.set_xlabel(r"Elapsed time $\mathit{t} - \mathit{t}_0$ (s)")
    ax_r.set_ylabel("Residual (hPa)")
    ax_p.set_title(f"{pd.Timestamp(row.datetime):%-d %b %H:%M}", fontsize=7)

    ax_b = fig.add_subplot(gs[:, 1])
    keys = list(MODELS)
    bp = ax_b.boxplot([rms[:, k] for k in range(len(keys))], orientation="horizontal", whis=(5, 95), showfliers=False,
                      widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.0))
    for patch, k in zip(bp["boxes"], keys):
        patch.set(facecolor=ps.DAY_COLORS[0] if k == "D" else "white", edgecolor=INK, alpha=0.6 if k == "D" else 1)
    ax_b.set_yticks(range(1, len(keys) + 1), [LABELS[k] for k in keys], fontsize=7)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, None)
    ax_b.set_xlabel("rms residual, first 2 s (hPa)")
    ax_b.set_title(f"All {len(rms)} cycles", fontsize=7)
    for ax, lab, x in ((ax_p, "a", -0.1), (ax_r, "b", -0.1), (ax_b, "c", -0.62)):
        ps.panel(ax, lab, x=x)
        tidy(ax)
    ps.save(fig, "fig_pressure")
    for k, key in enumerate(keys):
        print(f"{key}: median {np.median(rms[:, k]):.2f} hPa, 5-95 % {np.percentile(rms[:, k], 5):.2f}-"
              f"{np.percentile(rms[:, k], 95):.2f}")


if __name__ == "__main__":
    main()
