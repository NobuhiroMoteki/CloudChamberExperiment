# -*- coding: utf-8 -*-
"""Train the Tr-shape SBI posterior (cloudchamber/sbi_shape.py), check it on held-out simulations
and apply it to recorded decompressions (data/raw_YYYYMMDD/decomp_*.npz).

Usage: python run_sbi_shape.py [n_sims] [npz glob]
"""
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from cloudchamber import conversions as cv
from cloudchamber import sbi_shape as ss
from cloudchamber.shape_estimator import cycle_x

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_GLOB = os.path.join(HERE, os.pardir, "data", "raw_*", "decomp_*.npz")


def simulate(n_sims):
    path = os.path.join(MODEL_DIR, f"sbi_shape_sims_{n_sims}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["theta"], d["x"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        out = [r for r in pool.map(ss.simulate_one, range(n_sims), chunksize=64) if r is not None]
    theta = np.array([r[0] for r in out])
    x = np.array([r[1] for r in out])
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(path, theta=theta, x=x)
    print(f"simulated {n_sims} ({len(out)} usable) in {time.time()-t0:.0f} s", flush=True)
    return theta, x


def train(theta, x):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    prior = BoxUniform(low=torch.tensor(ss.THETA_LOW, dtype=torch.float32),
                       high=torch.tensor(ss.THETA_HIGH, dtype=torch.float32))
    inference = NPE(prior=prior, density_estimator="nsf")
    t0 = time.time()
    inference.append_simulations(torch.tensor(theta, dtype=torch.float32),
                                 torch.tensor(x, dtype=torch.float32)).train()
    posterior = inference.build_posterior()
    print(f"trained in {time.time()-t0:.0f} s", flush=True)
    return posterior


def summary(posterior, X, n=1000):
    s = posterior.sample_batched((n,), x=torch.tensor(np.atleast_2d(X), dtype=torch.float32),
                                 show_progress_bars=False).numpy()
    return np.median(s, 0), np.percentile(s, 5, 0), np.percentile(s, 95, 0)


def legacy_T_ini(d):
    """T_ini of files recorded before the 2026-10-29 thermistor fix (R0 = 10 kOhm, Rin = 1.75 MOhm)."""
    if "r0" in d.files:
        return float(d["T_ini"])
    v = cv.thermistor_to_volt(float(d["T_ini"]), 10e3, 1.75e6)
    return float(cv.thermistor_k(v, cv.R0_TH, np.inf))


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    pattern = sys.argv[2] if len(sys.argv) > 2 else DATA_GLOB
    theta, x = simulate(n_sims)
    n_test = 2000
    th_tr, x_tr, th_te, x_te = theta[:-n_test], x[:-n_test], theta[-n_test:], x[-n_test:]
    post_path = os.path.join(MODEL_DIR, f"npe_shape_{n_sims}.pt")
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

    print("\nrecorded cycles: SBI median [90% CI]  vs  hygrometer (Td output / RH output)")
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        T_ini = legacy_T_ini(d)
        X, info = cycle_x(d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float),
                          float(d["p_ini"]) * 1e2, T_ini)
        name = os.path.basename(f)[7:-4]
        if X is None:
            print(f"  {name}: no usable cloud ({info})")
            continue
        m, l, h = (a[0] for a in summary(posterior, X))
        p_ini = float(d["p_ini"]) * 1e2
        q_td = cv.qw_from_dewpoint(float(d["ref_Td"]), p_ini) * 1e3
        e = float(d["ref_RH"]) * 611.2 * np.exp(17.62 * (T_ini - 273.15) / (243.12 + T_ini - 273.15))
        q_rh = 0.622 * e / (p_ini - e) * 1e3
        print(f"  {name}: qw {m[0]:.2f} [{l[0]:.2f}, {h[0]:.2f}] g/kg  log10 N {m[1]:.2f} [{l[1]:.2f}, {h[1]:.2f}]"
              f"  | ref qw {q_td:.2f} / {q_rh:.2f}  | t_s {X[0]:.3f} s, p fit rms {info['p_rms']/100:.2f} hPa")


if __name__ == "__main__":
    main()
