# -*- coding: utf-8 -*-
"""Paper figure: two recorded cycles (low and high N) against time after the onset t0 of the pressure drop.

Top: measured transmittance with the forward model at the posterior median (leave-one-day-out estimate;
nominal nuisance parameters, sensor response P_GAIN_NOM, P_LAG_NOM) and the inference window
t_s - 0.1 ... t_s + 0.9 s. Middle and bottom: supersaturation and radius of the largest droplets of
the same simulation. The pressure of these cycles is shown in fig_pressure (paper_fig_pressure.py).

Usage: python paper_fig_cycle.py  -> Paper/JAerosolSci/figures/fig_cycle.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.activation import simulate_activation
from cloudchamber.sbi_shape import shape_features
from cloudchamber.sbi_shape2 import OUT_DT, SHAPE_OFFSETS, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from cloudchamber.shape_estimator import cycle_x
from fit_pressure_correction import NOMINAL, raw_path
from paper_fig_n_validation import DAYS, load
from paper_style import INK, INK2, tidy

TARGETS = (600.0, 8000.0)  # [cm-3] CPC of the cycles shown (closest cycle with N/N_CPC near 1)
T_MIN, T_MAX = 0.0, 3.0  # [s]


def pick(data):
    out = []
    for target in TARGETS:
        best = None
        for _, d, _, _ in data:
            d = d[(np.abs(d.nd_cm3 / d.cpc - 1) < 0.02)]
            if len(d):
                i = (np.abs(np.log(d.cpc / target))).idxmin()
                cand = d.loc[i]
                if best is None or abs(np.log(cand.cpc / target)) < abs(np.log(best.cpc / target)):
                    best = cand
        out.append(best)
    return out


def measured(row):
    d = np.load(raw_path(row.datetime))
    t, p, tr = d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float)
    p_ini, T_ini = float(d["p_ini"]) * 1e2, float(d["T_ini"])
    _, info = cycle_x(t, p, tr, p_ini, T_ini, offsets=SHAPE_OFFSETS)
    base = np.median(tr[(t < info["t0"]) & np.isfinite(tr)])
    return t - info["t0"], p, tr / base, p_ini, T_ini, info


def model(row, p_ini, T_ini, info):
    nom = {k: v for k, v in NOMINAL.items() if k != "forward"}
    cond = dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - ss4.P_GAIN_NOM * (p_ini - info["p_offset"]),
                tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])
    s = simulate_activation(row.qw_gkg * 1e-3, 10 ** row.log10nd, n_bins=12, t_end=T_END, out_dt=OUT_DT,
                            spread_exp=2.0, l_path=L_PATH3, forward=True, **nom, **cond)
    return dict(s, t=s["t"] - ss4.P_LAG_NOM)


def main():
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    rows = pick(data)
    ps.apply()
    fig, axs = plt.subplots(3, 2, figsize=(ps.ONE_HALF, 120 * ps.MM), sharex=True, sharey="row",
                            gridspec_kw=dict(height_ratios=[1.5, 1, 1]))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.92, hspace=0.15, wspace=0.08)
    for j, row in enumerate(rows):
        t, p, tr, p_ini, T_ini, info = measured(row)
        s = model(row, p_ini, T_ini, info)
        m = (t >= T_MIN) & (t <= T_MAX)
        mm = (s["t"] >= T_MIN) & (s["t"] <= T_MAX)
        f = shape_features(t, tr, onset_level=1 - 0.001, offsets=SHAPE_OFFSETS)
        for ax in axs[:, j]:
            if f is not None:
                ax.axvspan(f[0] + SHAPE_OFFSETS[0], f[0] + SHAPE_OFFSETS[-1], color=ps.GRID, alpha=0.8, lw=0,
                           label="Inference window" if ax is axs[0, j] else None)
        axs[0, j].plot(t[m], tr[m], color=INK, lw=0.6, label="Measured")
        axs[0, j].plot(s["t"][mm], s["Tr"][mm], color=ps.DAY_COLORS[0], lw=1.2, label="Model, posterior median")
        axs[1, j].axhline(0, color=INK2, lw=0.4)
        axs[1, j].plot(s["t"][mm], (s["S"][mm] - 1) * 100, color=ps.DAY_COLORS[0], lw=1.0)
        axs[2, j].plot(s["t"][mm], s["a_max"][mm] * 1e6, color=ps.DAY_COLORS[0], lw=1.0)
        axs[0, j].set_title(f"{pd.Timestamp(row.datetime):%-d %b %H:%M}:  $\\mathit{{N}}$ = {row.nd_cm3:.0f}, "
                            f"$\\mathit{{N}}_\\mathrm{{CPC}}$ = {row.cpc:.0f} cm$^{{-3}}$\n$\\mathit{{q}}_\\mathit{{w}}$ = {row.qw_gkg:.2f}, "
                            f"$\\mathit{{q}}_{{\\mathit{{w}},\\mathrm{{hyg}}}}$ = {row.qw_ref_td_gkg:.2f} g kg$^{{-1}}$", fontsize=7)
        axs[2, j].set_xlabel(r"Elapsed time $\mathit{t} - \mathit{t}_0$ (s)")
        print(f"{row.datetime}: N {row.nd_cm3:.0f} CPC {row.cpc:.0f}, qw {row.qw_gkg:.2f} hyg {row.qw_ref_td_gkg:.2f}, "
              f"t_s {f[0] if f is not None else np.nan:.2f} s, S_max-1 {(s['S_max'] - 1) * 100:.1f} %, "
              f"a_max at window end {np.interp(f[0] + SHAPE_OFFSETS[-1], s['t'], s['a_max']) * 1e6:.2f} um")
    axs[0, 0].set_ylabel("Transmittance Tr")
    axs[1, 0].set_ylabel("Supersaturation (%)")
    axs[2, 0].set_ylabel("Largest droplet\nradius (µm)")
    axs[1, 0].set_ylim(-5, None)
    axs[0, 0].legend(loc="lower left", frameon=False)
    axs[2, 0].set_xlim(T_MIN, T_MAX)
    axs[2, 0].set_xticks([0, 1, 2, 3])
    for ax, lab in zip(axs.flat, "abcdef"):
        ps.panel(ax, lab, x=-0.2 if ax in axs[:, 0] else -0.08)
        tidy(ax)
    ps.save(fig, "fig_cycle")


if __name__ == "__main__":
    main()
