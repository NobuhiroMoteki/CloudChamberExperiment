# -*- coding: utf-8 -*-
"""SI numbers: sensitivity of N to the nuisance parameters (uncertainty table of the paper, SI Sect. S9).

For cycles spread evenly over the hygrometer range (fit_pressure_correction.select), N is fitted with
qw fixed at the hygrometer value and the nominal pressure-sensor response of sbi_shape4 (gain
P_GAIN_NOM; uncorrected, i.e. 1, unless SBI4_PCORR=1), by minimising the misfit of
fit_pressure_correction, (dt_s / 20 ms)^2 + (rms dTr / 0.003)^2, over log10 N and over the lag
P_LAG_NOM +- LAG_HALF_NOM (without the correction a time shift within 2 s.d. of the t0 error dt0 of
the simulator; with it the lag within its prior).
This is repeated with one nuisance parameter at a time at an edge of its prior; the result is the
median of N(edge) / N(nominal) over the cycles. --fixed-lag keeps the lag at P_LAG_NOM.

Usage: python paper_si_nuisance.py [--n 104] [--csv ../data/reanalysis/reanalysis_nopcorr-cal1031-1101.csv]
  -> models/nuisance_sensitivity.csv (per cycle) and Paper/JAerosolSci/tables/tab_s_nuisance.tex
       python paper_si_nuisance.py --table-only   (table from the existing csv)
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
# (name, key, low edge, high edge) of the priors (Table 2 of the paper)
EDGES = [("alpha_c", "alpha_c", 0.1, 1.0), ("r_g", "r_g", 20e-9, 80e-9), ("sigma_g", "sigma_g", 1.4, 2.2),
         ("kappa", "kappa", 0.1, 0.7), ("spread", "sigma_rel", 0.015, 0.10),
         ("spread_floor", "spread_floor", 0.003, 0.03)]
TABLE = os.path.join(HERE, os.pardir, "Paper", "JAerosolSci", "tables", "tab_s_nuisance.tex")
LABELS = {"alpha_c": (r"Condensation coefficient $\alpha_c$", "0.1", "1"),
          "r_g": (r"Geometric mean dry radius $r_g$", "20\\,nm", "80\\,nm"),
          "sigma_g": (r"Geometric standard deviation $\sigma_g$", "1.4", "2.2"),
          "kappa": (r"Hygroscopicity $\kappa$", "0.1", "0.7"),
          "spread": ("Droplet-radius spread", "0.015", "0.10"),
          "spread_floor": ("Floor of the spread", "0.003", "0.03")}
DT_SCALE, TR_SCALE = 0.02, 0.003  # [s], [-] weights of the misfit (as for the gain-lag valley)
LAGS = P_LAG_NOM + np.arange(-LAG_HALF_NOM, LAG_HALF_NOM + 1e-9, 0.005)  # [s] scanned without new simulations


def variants():
    out = [("nominal", dict(NOMINAL))]
    for name, key, lo, hi in EDGES:
        out += [(f"{name}_lo", dict(NOMINAL, **{key: lo})), (f"{name}_hi", dict(NOMINAL, **{key: hi}))]
    return out


def fit_cycle(row, fixed_lag=False):
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
    qw = row["qw_ref_td_gkg"] * 1e-3
    cond = dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - P_GAIN_NOM * (p_ini - info["p_offset"]),
                tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])

    def cost(lgn, nom):
        s = simulate_activation(qw, 10 ** lgn, r_g=nom["r_g"], sigma_g=nom["sigma_g"], kappa=nom["kappa"],
                                alpha_c=nom["alpha_c"], n_bins=12, t_end=T_END, out_dt=OUT_DT,
                                sigma_rel=nom["sigma_rel"], spread_exp=2.0, spread_floor=nom["spread_floor"],
                                l_path=L_PATH3, forward=nom["forward"], **cond)
        trs = _blur(s["Tr"], BLUR, OUT_DT)
        best = 1e6
        for lag in ([P_LAG_NOM] if fixed_lag else LAGS):
            f = shape_features(s["t"] - lag, trs, onset_level=level, offsets=SHAPE_OFFSETS)
            if f is not None:
                best = min(best, ((f[0] - ts_m) / DT_SCALE) ** 2 + np.mean((f[1] - tr_m) ** 2) / TR_SCALE ** 2)
        return best

    out = dict(datetime=row["datetime"], qw_ref=row["qw_ref_td_gkg"], cpc=row.get("cpc_cm3", np.nan))
    lg0 = row["log10nd"]
    for name, nom in variants():
        r = minimize_scalar(cost, bounds=(lg0 - 0.4, lg0 + 0.4), args=(nom,), method="bounded",
                            options=dict(xatol=2e-3))
        out[name] = r.x
        out[name + "_cost"] = r.fun
    return out


def write_table(res):
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Parameter & Edge & $N/N_\mathrm{nominal}$, median (IQR) & Misfit, median \\", r"\midrule",
             f"Nominal set & -- & 1 & {res.nominal_cost.median():.2f} \\\\"]
    for name, _, _, _ in EDGES:
        label, lo, hi = LABELS[name]
        for e, v in (("lo", lo), ("hi", hi)):
            q = 10 ** (res[f"{name}_{e}"] - res.nominal)
            lines.append(f"{label if e == 'lo' else ''} & {v} & {q.median():.3f} ({q.quantile(.25):.3f}--"
                         f"{q.quantile(.75):.3f}) & {res[f'{name}_{e}_cost'].median():.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(TABLE, "w") as fh:
        fh.write("% Generated by Python/paper_si_nuisance.py\n" + "\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(TABLE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=104)
    ap.add_argument("--csv", default=os.path.join(HERE, os.pardir, "data", "reanalysis",
                                                  "reanalysis_nopcorr-cal1031-1101.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "nuisance_sensitivity.csv"))
    ap.add_argument("--fixed-lag", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.table_only:
        write_table(pd.read_csv(a.out))
        return
    df = pd.read_csv(a.csv)
    df = df[df.qw_gkg.notna() & df.qw_ref_td_gkg.notna() & (df.datetime >= "2026-10-31T12:18")]
    df = df[(df.datetime < "2026-11-02T10:00") | (df.datetime >= "2026-11-02T10:18")].sort_values("datetime")
    rows = select(df, a.n).to_dict("records")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = [r for r in pool.map(fit_cycle, rows, [a.fixed_lag] * len(rows)) if r is not None]
    res = pd.DataFrame(res)
    res.to_csv(a.out, index=False)
    print(f"{len(res)} cycles in {time.time()-t0:.0f} s -> {a.out}")
    for name, _, _, _ in EDGES:
        lo = 10 ** (res[f"{name}_lo"] - res.nominal)
        hi = 10 ** (res[f"{name}_hi"] - res.nominal)
        print(f"{name}: N(low edge)/N = {lo.median():.3f} (IQR {lo.quantile(.25):.3f}-{lo.quantile(.75):.3f}), "
              f"N(high edge)/N = {hi.median():.3f} (IQR {hi.quantile(.25):.3f}-{hi.quantile(.75):.3f})")
    write_table(res)


if __name__ == "__main__":
    main()
