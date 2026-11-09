# -*- coding: utf-8 -*-
"""Train the version-2 Tr-shape SBI posterior (cloudchamber/sbi_shape2.py: window to the second Mie
hump, narrowing droplet-size spread, effective optical path as a nuisance), check it on held-out
simulations and compare it with version 1 on recorded cycles against the chilled-mirror hygrometer.

Usage: python run_sbi_shape2.py [n_sims] [cloud_data csv]
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
from cloudchamber import sbi_shape2 as ss2
from cloudchamber.shape_estimator import cycle_x
from run_sbi_shape import summary, train

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA = os.path.join(HERE, os.pardir, "data")


def simulate(n_sims):
    path = os.path.join(MODEL_DIR, f"sbi_shape2_sims_{n_sims}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["theta"], d["x"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        out = [r for r in pool.map(ss2.simulate_one, range(n_sims), chunksize=64) if r is not None]
    theta = np.array([r[0] for r in out])
    x = np.array([r[1] for r in out])
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(path, theta=theta, x=x)
    print(f"simulated {n_sims} ({len(out)} usable) in {time.time()-t0:.0f} s", flush=True)
    return theta, x


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, "cloud_data_20261029_v3.csv")
    theta, x = simulate(n_sims)
    n_test = 2000
    th_tr, x_tr, th_te, x_te = theta[:-n_test], x[:-n_test], theta[-n_test:], x[-n_test:]
    post_path = os.path.join(MODEL_DIR, f"npe_shape2_{n_sims}.pt")
    if os.path.exists(post_path):
        posterior = torch.load(post_path, weights_only=False)
    else:
        posterior = train(th_tr, x_tr)
        torch.save(posterior, post_path)

    med, lo, hi = summary(posterior, x_te)
    rmse = np.sqrt(np.mean((med - th_te) ** 2, 0))
    cov = ((th_te >= lo) & (th_te <= hi)).mean(0)
    print(f"held-out simulations: rms qw {rmse[0]:.2f} g/kg, log10 N {rmse[1]:.3f}; "
          f"90% CI coverage qw {100*cov[0]:.0f}%, log10 N {100*cov[1]:.0f}%; median CI width "
          f"qw {np.median(hi[:, 0]-lo[:, 0]):.2f} g/kg, log10 N {np.median(hi[:, 1]-lo[:, 1]):.2f}")
    near = (th_te[:, 0] > 4) & (th_te[:, 0] < 8)
    rm = np.sqrt(np.mean((med[near] - th_te[near]) ** 2, 0))
    print(f"  qw 4-8 g/kg only: rms qw {rm[0]:.2f} g/kg, log10 N {rm[1]:.3f}")

    # recorded cycles: version 2 vs version 1 (the csv) vs hygrometer
    df = pd.read_csv(csv)
    df = df[(df.estimator == "ShapeSBIEstimator") & df.qw_gkg.notna()]
    df = df[df["flags"].isna() | (df["flags"] == "")]
    raw_dir = os.path.join(DATA, "raw_" + os.path.basename(csv)[11:19])
    npz = {round(float(np.load(f)["t_wall"]), 1): f for f in glob.glob(os.path.join(raw_dir, "*.npz"))}
    rows, X = [], []
    for _, r in df.iterrows():
        d = np.load(npz[round(r.Time_stamp, 1)])
        xx, _ = cycle_x(d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float),
                        float(d["p_ini"]) * 1e2, float(d["T_ini"]), offsets=ss2.SHAPE_OFFSETS)
        if xx is None:
            continue
        X.append(xx)
        rows.append(dict(datetime=r.datetime, qw_ref=cv.qw_from_dewpoint(r.ref_Td, r.p_ini * 1e2) * 1e3,
                         v1_qw=r.qw_gkg, v1_lo=r.qw_lo, v1_hi=r.qw_hi, v1_N=r.log10nd,
                         v1_Nlo=r.log10nd_lo, v1_Nhi=r.log10nd_hi, p_ini=r.p_ini, T_ini=r.T_ini))
    res = pd.DataFrame(rows)
    med, lo, hi = summary(posterior, np.array(X))
    res["v2_qw"], res["v2_lo"], res["v2_hi"] = med[:, 0], lo[:, 0], hi[:, 0]
    res["v2_N"], res["v2_Nlo"], res["v2_Nhi"] = med[:, 1], lo[:, 1], hi[:, 1]
    out = os.path.join(MODEL_DIR, f"compare_shape2_{os.path.basename(csv)[11:-4]}.csv")
    res.to_csv(out, index=False)

    print(f"\nrecorded cycles ({len(res)}): qw vs hygrometer (Td output), g/kg")
    for v in ("v1", "v2"):
        d = res[f"{v}_qw"] - res.qw_ref
        inside = ((res.qw_ref >= res[f"{v}_lo"]) & (res.qw_ref <= res[f"{v}_hi"])).mean()
        print(f"  {v}: bias {d.mean():+.2f}, sd {d.std():.2f}, corr {np.corrcoef(res[f'{v}_qw'], res.qw_ref)[0, 1]:.2f}, "
              f"median CI width {(res[f'{v}_hi'] - res[f'{v}_lo']).median():.2f}, ref inside 90% CI {100*inside:.0f}%")
        nd = cv.number_per_cm3(10 ** res[f"{v}_N"], res.p_ini * 1e2, res.T_ini)
        print(f"      log10 N median {res[f'{v}_N'].median():.2f} (= {np.median(nd):.0f} /cm3), "
              f"sd {res[f'{v}_N'].std():.3f}, median CI width {(res[f'{v}_Nhi'] - res[f'{v}_Nlo']).median():.2f}")
    print(f"  v1-v2 log10 N corr {np.corrcoef(res.v1_N, res.v2_N)[0, 1]:.2f}; saved {out}")


if __name__ == "__main__":
    main()
