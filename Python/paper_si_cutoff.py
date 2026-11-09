# -*- coding: utf-8 -*-
"""SI figure: smallest activated dry diameter (counting cut-off) from theory.

For every STEP-th cycle of the N validation, the forward model is run with the leave-one-day-out
estimates of qw and N, the recorded pressure curve (gain P_GAIN_NOM) and nominal nuisance parameters;
the peak supersaturation S_max and the temperature at the peak give, by kappa-Koehler theory, the
smallest dry diameter whose critical supersaturation is reached (kappa = 0: insoluble, wettable,
Kelvin equation). (a) S_max against N, (b) cut-off diameter against kappa (median and 5-95 % of the
cycles) with the result for lower S_max as a sensitivity, and the 7 nm of the TSI 3022A.

Usage: python paper_si_cutoff.py  -> Paper/JAerosolSci/figures/fig_s_cutoff.{pdf,png}
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import paper_style as ps
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.activation import _crit_radius, _seq, _sigma_w, simulate_activation
from cloudchamber.constants import RHO_L, RV
from cloudchamber.sbi_shape2 import OUT_DT, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from fit_pressure_correction import NOMINAL
from paper_fig_n_validation import DAYS, load
from paper_style import INK, INK2, tidy

STEP = 10
KAPPAS = np.round(np.arange(0.0, 0.71, 0.05), 2)
S_SENS = (0.05, 0.10)  # S_max - 1 for the sensitivity lines
CPC_D50 = 7.0  # [nm] TSI 3022A


def kelvin_a(T):
    return 2.0 * _sigma_w(T) / (RHO_L * RV * T)


def s_crit(rd, kappa, A):
    if kappa <= 0:
        return math.exp(A / rd)
    return _seq(_crit_radius(rd, kappa, A), rd ** 3, kappa, A)


def cutoff_diameter(s_max, T, kappa):
    """Smallest dry diameter [nm] with critical saturation ratio <= s_max."""
    A = kelvin_a(T)
    lo, hi = math.log(0.2e-9), math.log(1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if s_crit(math.exp(mid), kappa, A) > s_max:
            lo = mid
        else:
            hi = mid
    return 2e9 * math.exp(hi)


def run(r):
    p_ini = r.p_ini * 1e2
    nom = {k: v for k, v in NOMINAL.items() if k != "forward"}
    s = simulate_activation(r.qw_gkg * 1e-3, 10 ** r.log10nd, n_bins=12, t_end=T_END, out_dt=OUT_DT,
                            spread_exp=2.0, l_path=L_PATH3, forward=True, T_ini=r.T_ini, p_ini=p_ini,
                            p_offset=p_ini - ss4.P_GAIN_NOM * (p_ini - r.p_offset2 * 1e2), tau_vac=r.tau_vac1,
                            frac1=r.frac1, tau_vac2=r.tau_vac2, t_open=r.t_open, **nom)
    i = int(np.argmax(s["S"]))
    return s["S"][i], s["T"][i]


def main():
    d = pd.concat([load(f, a, b) for _, f, a, b, _, _ in DAYS]).iloc[::STEP]
    res = np.array([run(r) for r in d.itertuples()])
    s_max, T = res[:, 0], res[:, 1]
    dc = np.array([[cutoff_diameter(s, t, k) for k in KAPPAS] for s, t in zip(s_max, T)])
    print(f"{len(d)} cycles; S_max-1 {np.percentile(s_max-1, 5)*100:.0f}-{np.percentile(s_max-1, 95)*100:.0f} % "
          f"(median {np.median(s_max-1)*100:.0f} %), T at peak {T.min()-273.15:.1f}-{T.max()-273.15:.1f} C")
    for k in (0.0, 0.1, 0.3, 0.6):
        j = list(KAPPAS).index(k)
        print(f"  kappa {k}: cut-off {np.percentile(dc[:, j], 5):.1f}-{np.percentile(dc[:, j], 95):.1f} nm "
              f"(median {np.median(dc[:, j]):.1f})")
    ps.apply()
    fig, axs = plt.subplots(1, 2, figsize=(ps.ONE_HALF, 62 * ps.MM))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.92, wspace=0.35)
    ax = axs[0]
    for (lab, *_), col, mk in zip(DAYS, ps.DAY_COLORS, ps.DAY_MARKERS):
        m = d.datetime.str[:10] == {"31 Oct": "2026-10-31", "1 Nov": "2026-11-01", "2 Nov": "2026-11-02"}[lab]
        ax.scatter(d.nd_cm3[m], (s_max[m] - 1) * 100, s=6, marker=mk, color=col, lw=0, label=lab)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([300, 1000, 3000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\mathit{N}$ (cm$^{-3}$)")
    ax.set_ylabel("Peak supersaturation (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False)
    ps.panel(ax, "a", x=-0.25)
    tidy(ax)
    ax = axs[1]
    col = ps.DAY_COLORS[0]
    ax.fill_between(KAPPAS, np.percentile(dc, 5, 0), np.percentile(dc, 95, 0), color=col, alpha=0.3, lw=0,
                    label=r"Model $\mathit{S}_\mathrm{max}$, 5–95 %")
    ax.plot(KAPPAS, np.median(dc, 0), color=col, lw=1.2, label=r"Model $\mathit{S}_\mathrm{max}$, median")
    Tm = float(np.median(T))
    for s, ls in zip(S_SENS, ("--", ":")):
        ax.plot(KAPPAS, [cutoff_diameter(1 + s, Tm, k) for k in KAPPAS], color=INK2, lw=0.8, ls=ls,
                label=f"$\\mathit{{S}}_\\mathrm{{max}} - 1$ = {s*100:.0f} %")
    ax.axhline(CPC_D50, color=INK, lw=0.6)
    ax.text(0.68, CPC_D50 * 1.06, "CPC 3022A", ha="right", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, 0.7)
    ax.set_xlabel(r"Hygroscopicity $\mathit{\kappa}$")
    ax.set_ylabel("Cutoff dry diameter (nm)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ps.panel(ax, "b", x=-0.25)
    tidy(ax)
    ps.save(fig, "fig_s_cutoff")


if __name__ == "__main__":
    main()
