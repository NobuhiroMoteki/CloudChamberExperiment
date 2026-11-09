# -*- coding: utf-8 -*-
"""Re-fit the pressure-sensor correction of the SBI simulator (sbi_shape3.P_GAIN, P_LAG) on recorded
cycles with qw fixed at the chilled-mirror hygrometer value.

The true pressure drop is P_GAIN * the measured drop and the reading lags the flask by P_LAG.
For each selected cycle the simulator (nominal nuisances) is run on a grid of p_gain and log10 N
around the v3 estimate; p_lag is a pure time shift of the simulated Tr and is scanned without new
simulations. Stored per cycle: t_s(sim) - t_s(meas) and the rms Tr-shape difference on the
(gain, log10 N, lag) grid, so that the weighting can be chosen afterwards.

Usage: python fit_pressure_correction.py REANALYSIS.csv [REANALYSIS2.csv ...] [--n 240] [--out file.npz]
                                         [--alpha_c 0.3] [--kappa 0.3] [--no-forward]
                                         [--start 2026-10-31T12:18] [--end 2026-11-02T00:00]
v4 used models/pressure_correction_grid_fwd_ac{0.1,0.3,1.0}.npz (400 cycles, effective Qext).
  REANALYSIS.csv: output of reanalyze.py (selects the cycles and gives the v3 log10 N)
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cloudchamber.activation import simulate_activation
from cloudchamber.sbi_shape import _blur, shape_features
from cloudchamber.sbi_shape2 import OUT_DT, SHAPE_OFFSETS, T_END
from cloudchamber.sbi_shape3 import L_PATH3
from cloudchamber.shape_estimator import cycle_x

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
GAINS = np.round(np.arange(0.84, 1.051, 0.03), 3)
DLOGN = np.round(np.arange(-0.30, 0.301, 0.075), 3)
LAGS = np.round(np.arange(-0.10, 0.501, 0.01), 3)
NOMINAL = dict(r_g=40e-9, sigma_g=1.8, kappa=0.3, alpha_c=0.3, sigma_rel=0.04, spread_floor=0.01, forward=True)
BLUR = 0.02  # [s]


def raw_path(dt):
    s = dt[:19].replace("-", "").replace(":", "").replace("T", "_")
    return os.path.join(DATA, "raw_" + s[:8], f"decomp_{s}.npz")


def select(df, n, n_bins=8):
    """n cycles spread evenly over the hygrometer qw range (equal numbers per qw bin, evenly in time)."""
    edges = np.linspace(df.qw_ref_td_gkg.min(), df.qw_ref_td_gkg.max() + 1e-9, n_bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        g = df[(df.qw_ref_td_gkg >= a) & (df.qw_ref_td_gkg < b)]
        k = min(len(g), n // n_bins)
        out.append(g.iloc[np.linspace(0, len(g) - 1, k).round().astype(int)])
    return pd.concat(out)


def fit_cycle(row, nominal=NOMINAL):
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
    dts = np.full((GAINS.size, DLOGN.size, LAGS.size), np.nan)
    rms = np.full_like(dts, np.nan)
    for i, g in enumerate(GAINS):
        cond = dict(T_ini=T_ini, p_ini=p_ini, p_offset=p_ini - g * (p_ini - info["p_offset"]),
                    tau_vac=info["tau_vac"], frac1=info["frac1"], tau_vac2=info["tau_vac2"], t_open=info["t_open"])
        for j, dl in enumerate(DLOGN):
            s = simulate_activation(qw, 10 ** (row["log10nd"] + dl), r_g=nominal["r_g"], sigma_g=nominal["sigma_g"],
                                    kappa=nominal["kappa"], alpha_c=nominal["alpha_c"], n_bins=12, t_end=T_END,
                                    out_dt=OUT_DT, sigma_rel=nominal["sigma_rel"], spread_exp=2.0,
                                    spread_floor=nominal["spread_floor"], l_path=L_PATH3,
                                    forward=nominal["forward"], **cond)
            trs = _blur(s["Tr"], BLUR, OUT_DT)
            for k, lag in enumerate(LAGS):
                f = shape_features(s["t"] - lag, trs, onset_level=level, offsets=SHAPE_OFFSETS)
                if f is not None:
                    dts[i, j, k] = f[0] - ts_m
                    rms[i, j, k] = np.sqrt(np.mean((f[1] - tr_m) ** 2))
    return dict(datetime=row["datetime"], qw_ref=row["qw_ref_td_gkg"], log10nd_v3=row["log10nd"],
                ts_meas=ts_m, tr_min=float(tr_m.min()), dts=dts, rms=rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default=os.path.join(HERE, "models", "pressure_correction_grid.npz"))
    ap.add_argument("--alpha_c", type=float, default=NOMINAL["alpha_c"])
    ap.add_argument("--kappa", type=float, default=NOMINAL["kappa"])
    ap.add_argument("--no-forward", action="store_true", help="plain Qext (no forward-scattered light)")
    ap.add_argument("--start", default="", help="first cycle used, e.g. 2026-10-31T12:18")
    ap.add_argument("--end", default="9", help="cycles before this time are used")
    a = ap.parse_args()
    nominal = dict(NOMINAL, alpha_c=a.alpha_c, kappa=a.kappa, forward=not a.no_forward)
    df = pd.concat([pd.read_csv(f) for f in a.csv])
    df = df[df.qw_gkg.notna() & df.qw_ref_td_gkg.notna() & (df.datetime >= a.start) & (df.datetime < a.end)]
    df = df.sort_values("datetime")
    rows = select(df, a.n).to_dict("records")
    print(f"{len(rows)} cycles, qw_ref {min(r['qw_ref_td_gkg'] for r in rows):.2f}-"
          f"{max(r['qw_ref_td_gkg'] for r in rows):.2f} g/kg", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        res = []
        for k, r in enumerate(pool.map(fit_cycle, rows, [nominal] * len(rows))):
            if r is not None:
                res.append(r)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(rows)} ({time.time()-t0:.0f} s)", flush=True)
    np.savez(a.out, gains=GAINS, dlogn=DLOGN, lags=LAGS,
             datetime=np.array([r["datetime"] for r in res]),
             **{k: np.array([r[k] for r in res]) for k in ("qw_ref", "log10nd_v3", "ts_meas", "tr_min", "dts", "rms")})
    print(f"saved {a.out}: {len(res)} cycles in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
