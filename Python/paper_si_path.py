# -*- coding: utf-8 -*-
"""SI numbers: does a cloud-free layer that thickens with wall heating explain the curves better?

For the cycles of paper_si_nuisance.py (104, spread over the hygrometer range), log10 N is fitted with
qw fixed at the hygrometer value, nominal nuisance parameters and the lags of paper_si_nuisance, for
several treatments of the optical path L:
  - constant L = 0.18 m (the paper), 0.19 m, 0.17 m and 0.16 m;
  - L(t) = min(0.18 m, 0.197 m - 2 k sqrt(t)): a layer on each side growing by conduction, where
    k = c * 2 sqrt(alpha) with alpha = 2.4e-5 m2 s-1 and c = 1.163 (erfc(c) = 0.1: a tenth of the
    wall-core temperature difference) or half of that.
Reported: N relative to the paper's L, the total misfit (as in paper_si_nuisance.py), the rms Tr
residual in the second half of the inference window (printed only), and the mean residual (measured -
simulated Tr) at t_s + 1.2-1.5 s, after the window, where the measured Tr stops falling (Fig. S1); in
the table as the median over the lowest and the highest third of the cycles in N.

Usage: python paper_si_path.py [--n 104]  -> models/path_sensitivity.csv, Paper/JAerosolSci/tables/tab_s_path.tex
       python paper_si_path.py --table-only   (table from the existing csv)
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
from cloudchamber.sbi_shape4 import P_GAIN_NOM
from cloudchamber.shape_estimator import cycle_x
from fit_pressure_correction import BLUR, NOMINAL, raw_path, select
from paper_si_nuisance import DT_SCALE, LAGS, TR_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
D_INNER = 0.197  # [m] inner diameter of the flask
K_TENTH = 1.163 * 2 * np.sqrt(2.4e-5)  # [m s-1/2] layer thickness per sqrt(t), a tenth of the difference
CASES = [("L0.18", dict(l_path=L_PATH3)), ("L0.19", dict(l_path=0.19)), ("L0.17", dict(l_path=0.17)),
         ("L0.16", dict(l_path=0.16)),
         ("Lt_tenth", dict(l_path=L_PATH3, l_inner=D_INNER, l_wall_k=K_TENTH)),
         ("Lt_half", dict(l_path=L_PATH3, l_inner=D_INNER, l_wall_k=0.5 * K_TENTH))]
LATE = np.asarray(SHAPE_OFFSETS) > 0.45  # second half of the inference window
AFTER = np.arange(1.2, 1.5 + 1e-9, 0.01)  # [s] after t_s: after the window, where the measured Tr stops falling
TABLE = os.path.join(HERE, os.pardir, "Paper", "JAerosolSci", "tables", "tab_s_path.tex")
LABELS = {"L0.18": "0.18\\,m (paper)", "L0.19": "0.19\\,m", "L0.17": "0.17\\,m", "L0.16": "0.16\\,m",
          "Lt_tenth": "$L(t)$, $b$ for a tenth", "Lt_half": "$L(t)$, $b/2$"}


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
    after = shape_features(t - info["t0"], tr / base, onset_level=level, offsets=AFTER)
    if after is None:
        return None
    tr_after_m = after[1]
    qw = row["qw_ref_td_gkg"] * 1e-3
    cond = dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - P_GAIN_NOM * (p_ini - info["p_offset"]),
                tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])
    nom = NOMINAL

    def evaluate(lgn, optics):
        s = simulate_activation(qw, 10 ** lgn, r_g=nom["r_g"], sigma_g=nom["sigma_g"], kappa=nom["kappa"],
                                alpha_c=nom["alpha_c"], n_bins=12, t_end=T_END, out_dt=OUT_DT,
                                sigma_rel=nom["sigma_rel"], spread_exp=2.0, spread_floor=nom["spread_floor"],
                                forward=True, **optics, **cond)
        trs = _blur(s["Tr"], BLUR, OUT_DT)
        best = (1e6, np.nan, np.nan)
        for lag in LAGS:
            f = shape_features(s["t"] - lag, trs, onset_level=level, offsets=SHAPE_OFFSETS)
            if f is not None:
                c = ((f[0] - ts_m) / DT_SCALE) ** 2 + np.mean((f[1] - tr_m) ** 2) / TR_SCALE ** 2
                if c < best[0]:
                    g = shape_features(s["t"] - lag, trs, onset_level=level, offsets=AFTER)
                    best = (c, np.sqrt(np.mean((f[1][LATE] - tr_m[LATE]) ** 2)),
                            np.nan if g is None else np.mean(tr_after_m - g[1]))
        return best

    out = dict(datetime=row["datetime"], qw_ref=row["qw_ref_td_gkg"])
    lg0 = row["log10nd"]
    for name, optics in CASES:
        r = minimize_scalar(lambda lg: evaluate(lg, optics)[0], bounds=(lg0 - 0.4, lg0 + 0.4),
                            method="bounded", options=dict(xatol=2e-3))
        out[name] = r.x
        out[name + "_cost"] = r.fun
        _, out[name + "_late"], out[name + "_after"] = evaluate(r.x, optics)
    return out


def write_table(res):
    lg = res["L0.18"]  # log10 N [m-3] with the paper's path
    lo, hi = lg.quantile(1 / 3), lg.quantile(2 / 3)
    n_lo, n_hi = (float(f"{10 ** v * 1e-6:.2g}") for v in (lo, hi))  # [cm-3], two significant digits
    lines = [r"\begin{tabular}{llllll}", r"\toprule",
             r"Optical path & $N/N(0.18\,\mathrm{m})$ & Misfit & Better than & "
             r"\multicolumn{2}{l}{$\Delta$Tr after the window ($10^{-3}$)} \\",
             r"\cmidrule(l){5-6}",
             rf" & median (IQR) & median & 0.18\,m (\%) & $N < {n_lo:.0f}$\,cm$^{{-3}}$ & "
             rf"$N > {n_hi:.0f}$\,cm$^{{-3}}$ \\",
             r"\midrule"]
    for name, _ in CASES:
        q = 10 ** (res[name] - res["L0.18"])
        better = "--" if name == "L0.18" else f"{(res[name + '_cost'] < res['L0.18_cost']).mean() * 100:.0f}"
        a_lo, a_hi = (res.loc[m, name + "_after"].median() * 1e3 for m in (lg <= lo, lg >= hi))
        lines.append(f"{LABELS[name]} & {q.median():.3f} ({q.quantile(.25):.3f}--{q.quantile(.75):.3f}) & "
                     f"{res[name + '_cost'].median():.3f} & {better} & ${a_lo:+.1f}$ & ${a_hi:+.1f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(TABLE, "w") as fh:
        fh.write("% Generated by Python/paper_si_path.py\n" + "\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(TABLE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=104)
    ap.add_argument("--csv", default=os.path.join(HERE, os.pardir, "data", "reanalysis",
                                                  "reanalysis_nopcorr-cal1031-1101.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "path_sensitivity.csv"))
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
        res = pd.DataFrame([r for r in pool.map(fit_cycle, rows) if r is not None])
    res.to_csv(a.out, index=False)
    print(f"{len(res)} cycles in {time.time() - t0:.0f} s -> {a.out}")
    for name, _ in CASES:
        q = 10 ** (res[name] - res["L0.18"])
        better = (res[name + "_cost"] < res["L0.18_cost"]).mean()
        print(f"{name:9s} N/N(L=0.18) {q.median():.3f} (IQR {q.quantile(.25):.3f}-{q.quantile(.75):.3f}); "
              f"misfit median {res[name + '_cost'].median():.3f}, better than L=0.18 in {better * 100:.0f} %; "
              f"late rms dTr median {res[name + '_late'].median() * 1e3:.2f}e-3; after window "
              f"{res[name + '_after'].median() * 1e3:+.2f}e-3 (IQR {res[name + '_after'].quantile(.25) * 1e3:+.2f} to "
              f"{res[name + '_after'].quantile(.75) * 1e3:+.2f})")
    write_table(res)


if __name__ == "__main__":
    main()
