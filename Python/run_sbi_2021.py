# -*- coding: utf-8 -*-
"""Train the feature-based SBI posterior and compare it with linear regression,
on held-out simulations and on the 2021 campaign (chilled-mirror RH, CPC TSI 3022A).

Usage: python run_sbi_2021.py [n_sims]
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from cloudchamber import sbi_features as sf
from cloudchamber.constants import RD

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
B21 = os.path.join(HERE, "2021_data")
FILES = {f"May-{n}": os.path.join(B21, "KankyoJissyu2021_ObsData_1stGroups", f"obs_data_chamber_{n}.csv")
         for n in ("center", "left", "right")}
FILES.update({f"Jul-{n}": os.path.join(B21, "KankyoJissyu2021_ObsData_2ndGroups", f"obs_data_chamber_{n}.csv")
              for n in ("1", "2", "3")})


def simulate(n_sims):
    path = os.path.join(MODEL_DIR, f"sbi_features_sims_{n_sims}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["theta"], d["x"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        out = [r for r in pool.map(sf.simulate_one, range(n_sims), chunksize=32) if r is not None]
    theta = np.array([r[0] for r in out])
    x = np.array([r[1] for r in out])
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(path, theta=theta, x=x)
    print(f"simulated {n_sims} ({len(out)} with a cloud) in {time.time()-t0:.0f} s", flush=True)
    return theta, x


def train(theta, x):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    prior = BoxUniform(low=torch.tensor(sf.THETA_LOW, dtype=torch.float32),
                       high=torch.tensor(sf.THETA_HIGH, dtype=torch.float32))
    inference = NPE(prior=prior, density_estimator="nsf")
    t0 = time.time()
    inference.append_simulations(torch.tensor(theta, dtype=torch.float32),
                                 torch.tensor(x, dtype=torch.float32)).train()
    posterior = inference.build_posterior()
    print(f"trained in {time.time()-t0:.0f} s", flush=True)
    return posterior


def posterior_summary(posterior, X, n=1000):
    s = posterior.sample_batched((n,), x=torch.tensor(X, dtype=torch.float32),
                                 show_progress_bars=False).numpy()  # (n, batch, 2)
    return np.median(s, 0), np.percentile(s, 5, 0), np.percentile(s, 95, 0)


def ols(A, Y):
    return np.linalg.lstsq(A, Y, rcond=None)[0]


def design_course(x):  # Docs Sec. 5.2.2: [1, zeta_moist, zeta_turbid]
    return np.c_[np.ones(len(x)), x[:, 0], np.exp(x[:, 1])]


def design_full(x):  # same inputs as SBI
    return np.c_[np.ones(len(x)), x]


def load_2021(path, n=400, seed=0):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    T, p, poff, tau, pc, zt, rh, cpc = d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 7], d[:, 8], d[:, 9]
    ok = ((pc > 600) & (pc < 1000) & (zt > 0) & (rh > 0) & (cpc > 0) & ~np.isclose(p, 1013.25)
          & np.isfinite(rh + cpc))
    idx = np.nonzero(ok)[0]
    idx = np.random.default_rng(seed).choice(idx, min(n, idx.size), replace=False)
    T, p, poff, tau, pc, zt, rh, cpc = (a[idx] for a in (T, p, poff, tau, pc, zt, rh, cpc))
    X = np.array([sf.observed_x(1 - c / q, z, Ti, q, po, ta)
                  for c, z, Ti, q, po, ta in zip(pc, zt, T, p, poff, tau)])
    e = rh * 611.2 * np.exp(17.62 * (T - 273.15) / (243.12 + T - 273.15))  # RH relative to T_ini, Magnus
    qw_ref = 0.622 * e / (p * 1e2 - e) * 1e3
    lnN_ref = np.log10(cpc * 1e6 / (p * 1e2 / (RD * T)))
    return X, np.c_[qw_ref, lnN_ref]


def scores(est, ref, lo=None, hi=None):
    rel = (est[:, 0] - ref[:, 0]) / ref[:, 0]
    dn = est[:, 1] - ref[:, 1]
    s = (f"qw bias {100*np.median(rel):+5.1f}% rms {100*np.sqrt(np.mean(rel**2)):4.1f}% "
         f"r={np.corrcoef(est[:, 0], ref[:, 0])[0, 1]:.2f} | logN bias {np.median(dn):+5.2f} "
         f"rms {np.sqrt(np.mean(dn**2)):.2f} r={np.corrcoef(est[:, 1], ref[:, 1])[0, 1]:.2f}")
    if lo is not None:
        cov = ((ref >= lo) & (ref <= hi)).mean(0)
        s += f" | 90% CI coverage qw {100*cov[0]:.0f}% logN {100*cov[1]:.0f}%"
    return s


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    theta, x = simulate(n_sims)
    n_test = 2000
    th_tr, x_tr, th_te, x_te = theta[:-n_test], x[:-n_test], theta[-n_test:], x[-n_test:]

    post_path = os.path.join(MODEL_DIR, f"npe_features_{n_sims}.pt")
    if os.path.exists(post_path):
        posterior = torch.load(post_path, weights_only=False)
    else:
        posterior = train(th_tr, x_tr)
        torch.save(posterior, post_path)

    A_course = ols(design_course(x_tr), th_tr)
    A_full = ols(design_full(x_tr), th_tr)

    print("\n== held-out simulations (truth known), rms error: qw [g/kg] / log10 N")
    med, lo, hi = posterior_summary(posterior, x_te)
    for name, est in [("linear (course)", design_course(x_te) @ A_course),
                      ("linear (all inputs)", design_full(x_te) @ A_full), ("SBI median", med)]:
        rmse = np.sqrt(np.mean((est - th_te) ** 2, 0))
        print(f"  {name:20s} qw {rmse[0]:.2f}  logN {rmse[1]:.3f}")
    cov = ((th_te >= lo) & (th_te <= hi)).mean(0)
    print(f"  SBI 90% CI coverage: qw {100*cov[0]:.1f}%  logN {100*cov[1]:.1f}%   "
          f"median CI width: qw {np.median(hi[:,0]-lo[:,0]):.2f} g/kg, logN {np.median(hi[:,1]-lo[:,1]):.2f}")

    print("\n== 2021 campaign vs chilled mirror (qw) and CPC 3022A (log10 N)")
    for k, path in FILES.items():
        X, ref = load_2021(path)
        med, lo, hi = posterior_summary(posterior, X)
        print(f"{k}")
        print(f"   linear (course)     {scores(design_course(X) @ A_course, ref)}")
        print(f"   linear (all inputs) {scores(design_full(X) @ A_full, ref)}")
        print(f"   SBI median          {scores(med, ref, lo, hi)}", flush=True)


if __name__ == "__main__":
    main()
