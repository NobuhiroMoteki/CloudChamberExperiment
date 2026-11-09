# -*- coding: utf-8 -*-
"""Paper figure: why qw and N can both be inferred from the extinction curve.

Forward model (cloudchamber.activation, as in the v4 simulator with nominal nuisance parameters and
the effective Qext) for the recorded pressure curve of the high-N cycle of fig_cycle: transmittance Tr(t), supersaturation and
the radius of the largest droplets, (left) N varied at fixed qw, (right) qw varied at fixed N.
Time is from the start of the pressure drop (t0).

Usage: python paper_fig_sensitivity.py  -> Paper/JAerosolSci/figures/fig_sensitivity.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.activation import simulate_activation
from cloudchamber.sbi_shape2 import OUT_DT, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from fit_pressure_correction import NOMINAL
from paper_fig_cycle import measured, pick
from paper_fig_n_validation import DAYS, load
from paper_style import INK2, tidy

QW_REF, N_REF = 7.0, 2000.0  # [g/kg], [cm-3]
N_SET = (500.0, 2000.0, 8000.0)
QW_SET = (6.0, 7.0, 8.0)
T_MIN, T_MAX = 0.0, 3.0  # [s] shown, as in fig_cycle
COLORS = ps.DAY_COLORS


def pressure_curve():
    """Initial state and fitted pressure curve (Eq. 1) of the high-N cycle of fig_cycle, with the
    nominal pressure-sensor gain applied (true drop = gain x measured drop; 1 without the correction)."""
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    _, _, _, p_ini, T_ini, info = measured(pick(data)[1])
    return dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - ss4.P_GAIN_NOM * (p_ini - info["p_offset"]),
                tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])


def run(qw, n_cm3, cond):
    n_kg = n_cm3 * 1e6 / (cond["p_ini"] / (287.04 * cond["T_ini"]))
    nom = {k: v for k, v in NOMINAL.items() if k != "forward"}
    return simulate_activation(qw * 1e-3, n_kg, n_bins=12, t_end=T_END, out_dt=OUT_DT, spread_exp=2.0,
                               l_path=L_PATH3, forward=True, **nom, **cond)


def main():
    cond = pressure_curve()
    print("pressure curve:", {k: round(float(v), 3) for k, v in cond.items()})
    ps.apply()
    fig, axs = plt.subplots(3, 2, figsize=(ps.ONE_HALF, 125 * ps.MM), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.93, hspace=0.2, wspace=0.08)
    cases = [[(QW_REF, n, f"$\\mathit{{N}}$ = {n:.0f} cm$^{{-3}}$") for n in N_SET],
             [(q, N_REF, f"$\\mathit{{q}}_\\mathit{{w}}$ = {q:.0f} g kg$^{{-1}}$") for q in QW_SET]]
    for j, (title, cs) in enumerate(zip((f"$\\mathit{{q}}_\\mathit{{w}}$ = {QW_REF:.0f} g kg$^{{-1}}$, $\\mathit{{N}}$ varied",
                                         f"$\\mathit{{N}}$ = {N_REF:.0f} cm$^{{-3}}$, $\\mathit{{q}}_\\mathit{{w}}$ varied"), cases)):
        axs[0, j].set_title(title, fontsize=8)
        for (qw, n, lab), col in zip(cs, COLORS):
            s = run(qw, n, cond)
            s["t"] = s["t"] - ss4.P_LAG_NOM  # on the time axis of the recorded pressure, as in fig_cycle
            m = (s["t"] >= T_MIN) & (s["t"] <= T_MAX)
            axs[0, j].plot(s["t"][m], s["Tr"][m], color=col, label=lab)
            axs[1, j].plot(s["t"][m], (s["S"][m] - 1) * 100, color=col)
            axs[2, j].plot(s["t"][m], s["a_max"][m] * 1e6, color=col)
            print(f"  qw {qw} N {n}: S_max-1 {(s['S_max']-1)*100:.1f} %, Tr min {s['Tr'][m].min():.3f}")
        axs[0, j].legend(loc="lower left", frameon=False)
    axs[0, 0].set_ylabel("Transmittance Tr")
    axs[1, 0].set_ylabel("Supersaturation (%)")
    axs[2, 0].set_ylabel("Largest droplet\nradius (µm)")
    axs[1, 0].set_ylim(-5, None)
    for ax in axs[2]:
        ax.set_xlabel(r"Elapsed time $\mathit{t} - \mathit{t}_0$ (s)")
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_xticks([0, 1, 2, 3])
    for ax, lab in zip(axs.flat, "abcdef"):
        ax.axhline(0, color=INK2, lw=0.4) if ax in axs[1] else None
        ps.panel(ax, lab, x=-0.2 if ax in axs[:, 0] else -0.08)
        tidy(ax)
    ps.save(fig, "fig_sensitivity")


if __name__ == "__main__":
    main()
