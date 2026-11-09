# -*- coding: utf-8 -*-
"""Sensitivity of the fitted N to the prescribed water vapour mixing ratio, d ln N / d ln qw.

Same fits as paper_si_nuisance.py (nominal nuisance parameters, pressure-sensor response P_GAIN_NOM,
P_LAG_NOM +- LAG_HALF_NOM, misfit (dt_s / 20 ms)^2 + (rms dTr / 0.003)^2 minimised over log10 N), with qw fixed at
the hygrometer value times each factor in QW_FACTORS. For each cycle d ln N / d ln qw is the central
difference between the factors 1 - D and 1 + D; the median and IQR over the cycles are printed. This is
the basis of the statement "an error of 3 % in qw changes N by about X %" and of the hygrometer row of
the uncertainty table (paper_table_uncertainty.N_PER_QW).

Usage: python paper_qw_sensitivity.py [--n 104] [--csv ../data/reanalysis/reanalysis_nopcorr-cal1031-1101.csv]
  -> models/qw_sensitivity.csv (per cycle)
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cloudchamber.activation import simulate_activation
from cloudchamber.sbi_shape import _blur, shape_features
from cloudchamber.sbi_shape2 import OUT_DT, SHAPE_OFFSETS, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from cloudchamber.sbi_shape4 import LAG_HALF_NOM, P_GAIN_NOM, P_LAG_NOM
from cloudchamber.shape_estimator import cycle_x
from fit_pressure_correction import BLUR, NOMINAL, raw_path, select

HERE = os.path.dirname(os.path.abspath(__file__))
D = 0.03  # relative change of qw
QW_FACTORS = (1.0 - D, 1.0, 1.0 + D)
DT_SCALE, TR_SCALE = 0.02, 0.003  # [s], [-] weights of the misfit (as for the gain-lag valley)
LAGS = P_LAG_NOM + np.arange(-LAG_HALF_NOM, LAG_HALF_NOM + 1e-9, 0.005)  # [s] scanned without new simulations


def fit_cycle(row):
    d = np.load(raw_path(row["datetime"]))
    t, p, tr = d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float)
    p_ini, T_ini = float(d["p_ini"]) * 1e2, float(d["T_ini"])
    x, info = cycle_x(t, p, tr, p_ini, T_ini, offsets=SHAPE_OFFSETS)
    if x is None:
        return None
    pre = tr[(t < info["t0"]) & np.isfinite(tr)]
    base, sd = np.median(pre), np.std(pre)
    level = 1.0 - max(5 * sd / base, 0.001)
    meas = shape_features(t - info["t0"], tr / base, onset_level=level, offsets=SHAPE_OFFSETS)
    if meas is None:
        return None
    ts_m, tr_m = meas
    cond = dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - P_GAIN_NOM * (p_ini - info["p_offset"]),
                tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])
    nom = NOMINAL

    def cost(lgn, qw):
        s = simulate_activation(qw, 10 ** lgn, r_g=nom["r_g"], sigma_g=nom["sigma_g"], kappa=nom["kappa"],
                                alpha_c=nom["alpha_c"], n_bins=12, t_end=T_END, out_dt=OUT_DT,
                                sigma_rel=nom["sigma_rel"], spread_exp=2.0, spread_floor=nom["spread_floor"],
                                l_path=L_PATH3, forward=nom["forward"], **cond)
        trs = _blur(s["Tr"], BLUR, OUT_DT)
        best = 1e6
        for lag in LAGS:
            f = shape_features(s["t"] - lag, trs, onset_level=level, offsets=SHAPE_OFFSETS)
            if f is not None:
                best = min(best, ((f[0] - ts_m) / DT_SCALE) ** 2 + np.mean((f[1] - tr_m) ** 2) / TR_SCALE ** 2)
        return best

    out = dict(datetime=row["datetime"], qw_ref=row["qw_ref_td_gkg"])
    lg0 = row["log10nd"]
    for k in QW_FACTORS:
        r = minimize_scalar(cost, bounds=(lg0 - 0.5, lg0 + 0.5), args=(row["qw_ref_td_gkg"] * 1e-3 * k,),
                            method="bounded", options=dict(xatol=1e-3))
        out[f"log10n_{k:.2f}"] = r.x
        out[f"cost_{k:.2f}"] = r.fun
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=104)
    ap.add_argument("--csv", default=os.path.join(HERE, os.pardir, "data", "reanalysis",
                                                  "reanalysis_nopcorr-cal1031-1101.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "qw_sensitivity.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df = df[df.qw_gkg.notna() & df.qw_ref_td_gkg.notna() & (df.datetime >= "2026-10-31T12:18")]
    df = df[(df.datetime < "2026-11-02T10:00") | (df.datetime >= "2026-11-02T10:18")].sort_values("datetime")
    rows = select(df, a.n).to_dict("records")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = pd.DataFrame([r for r in pool.map(fit_cycle, rows) if r is not None])
    res.to_csv(a.out, index=False)
    lo, mid, hi = (f"log10n_{k:.2f}" for k in QW_FACTORS)
    slope = (res[hi] - res[lo]) * np.log(10) / (np.log(1 + D) - np.log(1 - D))  # d ln N / d ln qw
    up, down = 10 ** (res[hi] - res[mid]) - 1, 10 ** (res[lo] - res[mid]) - 1
    print(f"{len(res)} cycles in {time.time()-t0:.0f} s -> {a.out}")
    print(f"d ln N / d ln qw: median {slope.median():.2f} (IQR {slope.quantile(.25):.2f}-{slope.quantile(.75):.2f})")
    print(f"qw +{D*100:.0f} %: N {up.median()*100:+.1f} % (IQR {up.quantile(.25)*100:+.1f} to {up.quantile(.75)*100:+.1f}); "
          f"qw -{D*100:.0f} %: N {down.median()*100:+.1f} % (IQR {down.quantile(.25)*100:+.1f} to {down.quantile(.75)*100:+.1f})")
    print(f"misfit median: x0.97 {res['cost_0.97'].median():.2f}, x1.00 {res['cost_1.00'].median():.2f}, "
          f"x1.03 {res['cost_1.03'].median():.2f}")


if __name__ == "__main__":
    main()
