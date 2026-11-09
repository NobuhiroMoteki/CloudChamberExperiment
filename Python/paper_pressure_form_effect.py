# -*- coding: utf-8 -*-
"""Effect of the form of the pressure curve on the estimates (paper Sect. 2.4).

For recorded cycles, the transmittance simulated with the fit of Eq. (1) and the posterior-median
(qw, N) (leave-one-day-out estimates; nominal nuisance parameters, sensor gain P_GAIN_NOM) is taken as
the truth. The same cycle is then analysed with the pressure curve replaced by the fit of a single
exponential with instantaneous opening (paper_pressure_models.py, model A, on the time axis of Eq. 1):
(qw, N) are refitted by minimising the rms difference of Tr over the inference window of the truth.
The ratios qw_A/qw and N_A/N show the bias that the single exponential would cause.
Cycles: the two of fig_cycle and N_EVEN cycles spread evenly over the evaluation. A second test uses
the pressure curves and initial states of the two fig_cycle cycles with synthetic (qw, N) up to
QW_SCAN[-1] (humid air, in which the cloud forms while the valve opens).

Usage: python paper_pressure_form_effect.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cloudchamber import sbi_shape4 as ss4
from cloudchamber.activation import simulate_activation
from cloudchamber.features import despike, find_decompression_start
from cloudchamber.sbi_shape import shape_features
from cloudchamber.sbi_shape2 import OUT_DT, SHAPE_OFFSETS, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from fit_pressure_correction import NOMINAL
from paper_fig_cycle import measured, pick
from paper_fig_n_validation import DAYS, load
from paper_pressure_models import DP, MODELS
from paper_pressure_models import fit as fit_pressure_model

N_EVEN = 30
QW_SCAN = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0)  # [g/kg] synthetic, beyond the evaluated 5.8-9.2
N_SCAN = (1000.0, 5000.0)  # [cm-3]


def sim(qw, n_kg, cond, shift=0.0):
    """Tr(t) [-] on the time axis of Eq. (1); `shift` [s] is the onset of the pressure curve on it."""
    nom = {k: v for k, v in NOMINAL.items() if k != "forward"}
    s = simulate_activation(qw, n_kg, n_bins=12, t_end=T_END, out_dt=OUT_DT, spread_exp=2.0, l_path=L_PATH3,
                            forward=True, **nom, **cond)
    return s["t"] + shift, s["Tr"]


def one(row):
    t, p, tr, p_ini, T_ini, info = measured(row)
    g = ss4.P_GAIN_NOM  # true drop = g x measured drop

    def cond(p_off, tau1, f1, tau2, t_open):
        return dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - g * (p_ini - p_off), tau_vac=tau1, frac1=f1,
                    tau_vac2=tau2, t_open=t_open)

    c1 = cond(info["p_offset"], info["tau_vac"], info["frac1"], info["tau_vac2"], info["t_open"])
    pd_ = despike(p)
    par, _, _ = fit_pressure_model(t, pd_, p_ini, find_decompression_start(pd_, p_ini, DP), MODELS["A"])
    t0_a, p_off_a, tau_a = par[0], par[1], par[2]
    ca = cond(p_off_a, tau_a, 1.0, 10.0, 0.0)

    qw0, n0 = row.qw_gkg * 1e-3, 10 ** row.log10nd
    tt, tr_true = sim(qw0, n0, c1)
    f = shape_features(tt, tr_true, onset_level=1 - 0.001, offsets=SHAPE_OFFSETS)
    if f is None:
        return None
    w = (tt >= f[0] + SHAPE_OFFSETS[0]) & (tt <= f[0] + SHAPE_OFFSETS[-1])

    def cost(x):
        ta, tra = sim(qw0 * np.exp(x[0]), n0 * np.exp(x[1]), ca, shift=t0_a)
        return float(np.sqrt(np.mean((np.interp(tt[w], ta, tra) - tr_true[w]) ** 2)))

    r = minimize(cost, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=200))
    return dict(datetime=row.datetime, N=row.nd_cm3, qw=row.qw_gkg, t_s=float(f[0]), qw_ratio=float(np.exp(r.x[0])),
                N_ratio=float(np.exp(r.x[1])), rms_Tr=r.fun, rms_Tr_unfitted=cost([0.0, 0.0]))


def synthetic(args):
    """one() for a recorded cycle with its (qw, N) replaced by synthetic values [g/kg], [cm-3]."""
    row, qw, n_cm3 = args
    r = row.copy()
    r["qw_gkg"] = qw
    r["log10nd"] = row["log10nd"] + np.log10(n_cm3 / row["nd_cm3"])  # per kg, same conversion as the row
    r["nd_cm3"] = n_cm3
    return one(r)


def main():
    data = [(lab, load(f, a, b), col, mk) for lab, f, a, b, col, mk in DAYS]
    rows = pick(data)
    a = pd.concat([d for _, d, _, _ in data]).dropna(subset=["qw_gkg", "log10nd"]).sort_values("datetime")
    rows += [a.iloc[i] for i in np.linspace(0, len(a) - 1, N_EVEN).astype(int)]
    with Pool() as pool:
        res = [r for r in pool.map(one, rows) if r is not None]
    df = pd.DataFrame(res)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    e = df.iloc[2:]
    for k in ("qw_ratio", "N_ratio"):
        print(f"{k} over {len(e)} even cycles: median {e[k].median():.3f}, range {e[k].min():.3f}-{e[k].max():.3f}")
    jobs = [(row, qw, n) for qw in QW_SCAN for row in rows[:2] for n in N_SCAN]
    with Pool() as pool:
        syn = pd.DataFrame([r for r in pool.map(synthetic, jobs) if r is not None])
    print("\nsynthetic qw (pressure curves of the two fig_cycle cycles):")
    for qw, g in syn.groupby("qw"):
        print(f"  qw {qw:4.1f}: t_s {g.t_s.min():.2f}-{g.t_s.max():.2f} s, qw_ratio {g.qw_ratio.min():.3f}-"
              f"{g.qw_ratio.max():.3f}, N_ratio {g.N_ratio.min():.3f}-{g.N_ratio.max():.3f}, "
              f"rms_Tr after refit <= {g.rms_Tr.max():.4f}")


if __name__ == "__main__":
    main()
