# -*- coding: utf-8 -*-
"""Paper table: uncertainty budget of N and the role of the forward-scattering correction.

Ablation: a model trained like the time-split model (pressure-curve library of 31 Oct-1 Nov, tag
nopcorr-cal1031-1101) without the forward-scatter correction (run_sbi_shape4.py --tag nopcorr-abl-nofwd
with SBI4_FORWARD=0), evaluated on 2 Nov. No model uses the pressure-sensor correction (SBI4_PCORR=0). Model variance: leave-one-day-out models vs the
all-period model on the same cycles. Nuisance parameters: best-fit N with one nuisance at the edge
of its prior (qw fixed at the hygrometer, uncorrected pressure), relative to the nominal set,
median over the cycles of models/nuisance_sensitivity.csv (paper_si_nuisance.py, 104 cycles).

Usage: python paper_table_uncertainty.py  -> Paper/JAerosolSci/tables/tab_uncertainty.tex
"""
import os

import numpy as np
import pandas as pd

from paper_fig_n_validation import DAYS, load
from paper_inlet_losses import ratio_factor_range
from paper_style import HERE

OUT = os.path.join(HERE, os.pardir, "Paper", "JAerosolSci", "tables", "tab_uncertainty.tex")
ABLATIONS = [("Full model", "reanalysis_nopcorr-cal1031-1101.csv"),
             ("Without the forward-scattering correction", "reanalysis_nopcorr-abl-nofwd.csv")]
DAY2 = ("2026-11-02", "2026-11-03")
# columns of models/nuisance_sensitivity.csv (log10 N at the low and high prior edges)
NUISANCE = [(r"Condensation coefficient $\alpha_c$ (0.1, 1)", "alpha_c"),
            (r"Geometric mean dry radius $r_g$ (20, 80\,nm)", "r_g"),
            (r"Geometric standard deviation $\sigma_g$ (1.4, 2.2)", "sigma_g"),
            (r"Hygroscopicity $\kappa$ (0.1, 0.7)", "kappa"),
            (r"Droplet-radius spread (0.015, 0.10)", "spread"),
            (r"Floor of the spread (0.003, 0.03)", "spread_floor")]
SENS = os.path.join(HERE, "models", "nuisance_sensitivity.csv")
CPC_UNC = 0.05  # two CPCs agree within 5 %
INPUT = os.path.join(HERE, "models", "input_sensitivity.csv")  # paper_input_sensitivity.py
DT_INI, DP_INI, DS_DROP = 0.1, 3.4, 0.7  # [K], [hPa], [%] errors of T_ini, p_ini and the pressure drop
# (thermocouple accuracy; Setra 276 accuracy 0.25 % FS; as in paper_input_sensitivity.py)
PATH = os.path.join(HERE, "models", "path_sensitivity.csv")  # fits with L = 0.17, 0.19 m (paper_si_path.py)


def pct(f):
    """Relative change of a ratio f as signed per cent, without a sign for zero."""
    v = round((f - 1) * 100, 1)
    return "0.0\\,\\%" if v == 0 else f"${v:+.1f}$\\,\\%"


def ratios(fname, start, end):
    d = load(fname, start, end)
    return d.nd_cm3 / d.cpc, d.qw_gkg / d.qw_ref_td_gkg


def main():
    rows = [r"\begin{tabular}{lll}", r"\toprule", r"Element & Effect on $N$ & Basis \\", r"\midrule",
            r"\multicolumn{3}{l}{\textit{Forward-scattering correction (2 Nov, estimator trained on 31 Oct--1 Nov)}} \\"]
    for name, f in ABLATIONS:
        if not os.path.exists(os.path.join(HERE, os.pardir, "data", "reanalysis", f)):
            print("missing", f)
            continue
        r, q = ratios(f, *DAY2)
        rows.append(f"{name} & $N/N_\\mathrm{{CPC}}$ = {r.median():.3f} ($q_w/q_{{w,\\mathrm{{hyg}}}}$ = {q.median():.3f}) & "
                    f"comparison on 2 Nov \\\\")
        print(f"{name}: N/CPC {r.median():.3f} (IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}), qw/hyg {q.median():.3f}")
    rows.append(r"\multicolumn{3}{l}{\textit{Nuisance parameters at the edges of their priors}} \\")
    sens = pd.read_csv(SENS)
    comb = []  # relative contributions combined in quadrature (all but alpha_c and the inlet losses)
    for name, key in NUISANCE:
        lo, hi = (np.median(10 ** (sens[f"{key}_{e}"] - sens.nominal)) for e in ("lo", "hi"))
        if key != "alpha_c":
            comb.append(max(abs(lo - 1), abs(hi - 1)))
        rows.append(f"{name} & {pct(lo)}, {pct(hi)} & sensitivity fits (Sect.~\\ref{{S-sec:s-nuisance}}) \\\\")
    # model variance: leave-one-day-out models vs the all-period model on the same cycles
    all_period = pd.read_csv(os.path.join(HERE, os.pardir, "data", "reanalysis", "reanalysis_nopcorr-all.csv"))
    m = []
    for _, f, a, b, _, _ in DAYS:
        d = load(f, a, b).merge(all_period[["datetime", "nd_cm3"]], on="datetime", suffixes=("", "_all"))
        m.append(d.nd_cm3 / d.nd_cm3_all)
    m = pd.concat(m)
    f_lo, f_hi = ratio_factor_range()
    ps = pd.read_csv(PATH)
    l_hi, l_lo = (np.median(10 ** (ps[c] - ps["L0.18"])) for c in ("L0.19", "L0.17"))
    inp = pd.read_csv(INPUT, comment="#").set_index("shift")
    t_n = (abs(inp.N_ratio["T+"] - 1) + abs(inp.N_ratio["T-"] - 1)) / 2
    t_q = (abs(inp.qw_ratio["T+"] - 1) + abs(inp.qw_ratio["T-"] - 1)) / 2
    p_n = (abs(inp.N_ratio["p+"] - 1) + abs(inp.N_ratio["p-"] - 1)) / 2
    s_n = (abs(inp.N_ratio["s+"] - 1) + abs(inp.N_ratio["s-"] - 1)) / 2
    s_q = (abs(inp.qw_ratio["s+"] - 1) + abs(inp.qw_ratio["s-"] - 1)) / 2
    comb += [abs(m.median() - 1), (abs(l_hi - 1) + abs(l_lo - 1)) / 2, t_n, p_n, s_n, CPC_UNC]
    total = np.sqrt(np.sum(np.square(comb)))
    rows += [r"\multicolumn{3}{l}{\textit{Other}} \\",
             f"Days used to train the estimator & {abs(m.median()-1)*100:.1f}\\,\\% (IQR ${(m.quantile(.25)-1)*100:+.1f}$ to "
             f"${(m.quantile(.75)-1)*100:+.1f}$\\,\\%) & leave-one-day-out vs all-day estimator \\\\",
             f"Optical path $L = 0.18 \\pm 0.01$\\,m & {pct(l_hi)} (0.19\\,m), {pct(l_lo)} (0.17\\,m) & "
             r"fits with the other path (Sect.~\ref{S-sec:s-thermo}) \\",
             f"Inlet losses, CPC vs chamber & ${(f_lo-1)*100:+.1f}$ to ${(f_hi-1)*100:+.1f}$\\,\\% in $N/N_\\mathrm{{CPC}}$ & "
             r"diffusion in the inlets (Sect.~\ref{sec:comparability}) \\",
             f"Initial temperature $T_\\mathrm{{ini}}$ ($\\pm{DT_INI}$\\,K) & $\\mp{t_n*100:.1f}$\\,\\% "
             f"($q_w$ $\\pm{t_q*100:.1f}$\\,\\%) & estimator with shifted input \\\\",
             f"Initial pressure $p_\\mathrm{{ini}}$ ($\\pm{DP_INI}$\\,hPa) & $\\pm{p_n*100:.1f}$\\,\\% & "
             r"estimator with shifted input \\",
             f"Pressure drop ($\\pm{DS_DROP}$\\,\\%, span and non-linearity) & $\\pm{s_n*100:.1f}$\\,\\% "
             f"($q_w$ $\\mp{s_q*100:.1f}$\\,\\%) & estimator with scaled input \\\\",
             f"Reference CPC & $\\pm{CPC_UNC*100:.0f}$\\,\\% & two units compared \\\\",
             r"\midrule",
             f"Combined, without $\\alpha_c$ and inlet losses & $\\approx\\pm{total*100:.0f}$\\,\\% & "
             r"in quadrature \\",
             r"\bottomrule", r"\end{tabular}"]
    print(f"model variance: median {m.median():.4f}, IQR {m.quantile(.25):.4f}-{m.quantile(.75):.4f}; "
          f"combined {total*100:.1f} % from {[round(c*100, 1) for c in comb]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("% generated by Python/paper_table_uncertainty.py\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
