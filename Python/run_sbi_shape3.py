# -*- coding: utf-8 -*-
"""Train the version-3 Tr-shape SBI posterior (cloudchamber/sbi_shape3.py: version 2 with corrected
thermodynamics, the pressure-sensor correction, no wall heating and L = 0.18 m) and check it on
held-out simulations and on recorded cycles against the chilled-mirror hygrometer and the CPC.

Usage: python run_sbi_shape3.py [n_sims]
"""
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from cloudchamber import conversions as cv
from cloudchamber import cpc
from cloudchamber import sbi_shape3 as ss3
from cloudchamber.shape_estimator import cycle_x
from run_sbi_shape import summary, train

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA = os.path.join(HERE, os.pardir, "data")


def simulate(n_sims):
    path = os.path.join(MODEL_DIR, f"sbi_shape3_sims_{n_sims}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["theta"], d["x"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        out = [r for r in pool.map(ss3.simulate_one, range(n_sims), chunksize=64) if r is not None]
    theta = np.array([r[0] for r in out])
    x = np.array([r[1] for r in out])
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(path, theta=theta, x=x)
    print(f"simulated {n_sims} ({len(out)} usable) in {time.time()-t0:.0f} s", flush=True)
    return theta, x


def recorded_cycles():
    """x of every recorded cycle with the hygrometer and the smoothed CPC (AIM exports) at its time."""
    rows, X = [], []
    for csv in sorted(glob.glob(os.path.join(DATA, "cloud_data_*.csv"))):
        df = pd.read_csv(csv)
        if "flags" in df:
            df = df[df["flags"].isna() | (df["flags"] == "")]
        raw_dir = os.path.join(DATA, "raw_" + os.path.basename(csv)[11:19])
        npz = {round(float(np.load(f)["t_wall"]), 1): f for f in glob.glob(os.path.join(raw_dir, "*.npz"))}
        for _, r in df.iterrows():
            f = npz.get(round(r.Time_stamp, 1))
            if f is None or not np.isfinite(r.ref_Td):
                continue
            d = np.load(f)
            xx, _ = cycle_x(d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float),
                            float(d["p_ini"]) * 1e2, float(d["T_ini"]), offsets=ss3.SHAPE_OFFSETS)
            if xx is None:
                continue
            X.append(xx)
            rows.append(dict(datetime=r.datetime, qw_ref=cv.qw_from_dewpoint(r.ref_Td, r.p_ini * 1e2) * 1e3,
                             p_ini=r.p_ini, T_ini=r.T_ini))
    res = pd.DataFrame(rows)
    aim = sorted(glob.glob(os.path.join(HERE, os.pardir, "CPC3022A", "data", "*.csv")))
    if aim:
        c = cpc.smooth(pd.concat([cpc.read_aim_csv(p) for p in aim]).sort_index())
        times = pd.to_datetime(res.datetime.str[:19])
        res["cpc"] = cpc.at_times(c, times)
    return res, np.array(X)


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    theta, x = simulate(n_sims)
    n_test = 2000
    th_tr, x_tr, th_te, x_te = theta[:-n_test], x[:-n_test], theta[-n_test:], x[-n_test:]
    post_path = os.path.join(MODEL_DIR, f"npe_shape3_{n_sims}.pt")
    if os.path.exists(post_path):
        posterior = torch.load(post_path, weights_only=False)
    else:
        t0 = time.time()
        posterior = train(th_tr, x_tr)
        torch.save(posterior, post_path)
        print(f"trained in {time.time()-t0:.0f} s", flush=True)

    med, lo, hi = summary(posterior, x_te)
    rmse = np.sqrt(np.mean((med - th_te) ** 2, 0))
    cov = ((th_te >= lo) & (th_te <= hi)).mean(0)
    print(f"held-out simulations: rms qw {rmse[0]:.2f} g/kg, log10 N {rmse[1]:.3f}; "
          f"90% CI coverage qw {100*cov[0]:.0f}%, log10 N {100*cov[1]:.0f}%")

    res, X = recorded_cycles()
    med, lo, hi = summary(posterior, X)
    res["v3_qw"], res["v3_lo"], res["v3_hi"] = med[:, 0], lo[:, 0], hi[:, 0]
    res["v3_N"] = cv.number_per_cm3(10 ** med[:, 1], res.p_ini * 1e2, res.T_ini)
    out = os.path.join(MODEL_DIR, "compare_shape3.csv")
    res.to_csv(out, index=False)
    r = res.v3_qw / res.qw_ref
    print(f"\nrecorded cycles ({len(res)}): qw / hygrometer median {r.median():.3f} "
          f"(IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}), within 10% {100*np.mean(abs(r-1) < .1):.0f}%, "
          f"bias {np.mean(res.v3_qw - res.qw_ref):+.2f} g/kg, corr {np.corrcoef(res.v3_qw, res.qw_ref)[0, 1]:.2f}")
    if "cpc" in res:
        ok = res.cpc.notna()
        rn = res.v3_N[ok] / res.cpc[ok]
        print(f"  N / CPC ({ok.sum()} cycles) median {rn.median():.3f} (IQR {rn.quantile(.25):.3f}-"
              f"{rn.quantile(.75):.3f}), corr {np.corrcoef(res.v3_N[ok], res.cpc[ok])[0, 1]:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
