# -*- coding: utf-8 -*-
"""Train the version-4 Tr-shape SBI posterior (cloudchamber/sbi_shape4.py: version 3 with the
pressure curve given by its drop at fixed times and the re-fitted pressure-sensor correction) and
check it on held-out simulations. Recorded cycles are re-estimated with reanalyze.py.

Usage: python run_sbi_shape4.py [n_sims] [--csv REANALYSIS.csv ...] [--start T] [--end T]
                                [--tag NAME --lag-ref 0.07]
  --csv (output of reanalyze.py): cycles whose pressure fits form the library of pressure curves;
    needed only when the library does not exist yet; --start/--end restrict them in time.
  --tag: variant, e.g. calibrated on one period only. Its library, simulations and posterior get
    the tag in their names (models/ramp2_library_TAG.npz, npe_shape4_TAG_N.pt), so the GUI keeps
    loading the main model; --lag-ref sets sbi_shape4.P_LAG_REF for it.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def build_library(csvs, path, start="", end="9"):
    """Pressure fits of the recorded cycles (columns of reanalyze.py; pressures in hPa)."""
    df = pd.concat([pd.read_csv(f) for f in csvs])
    df = df[df.t0_ramp2.notna() & df.p_offset2.notna() & (df.datetime >= start) & (df.datetime < end)]
    lib = dict(p_offset_rel=(df.p_offset2 / df.p_ini).to_numpy(), tau_vac=df.tau_vac1.to_numpy(),
               frac1=df.frac1.to_numpy(), tau_vac2=df.tau_vac2.to_numpy(), t_open=df.t_open.to_numpy())
    np.savez(path, **lib)
    print(f"pressure-curve library: {len(df)} cycles -> {path}", flush=True)


def simulate(ss4, n_sims, tag):
    path = os.path.join(MODEL_DIR, f"sbi_shape4{tag}_sims_{n_sims}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["theta"], d["x"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=11) as pool:
        out = [r for r in pool.map(ss4.simulate_one, range(n_sims), chunksize=64) if r is not None]
    theta = np.array([r[0] for r in out])
    x = np.array([r[1] for r in out])
    np.savez(path, theta=theta, x=x)
    print(f"simulated {n_sims} ({len(out)} usable) in {time.time()-t0:.0f} s", flush=True)
    return theta, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_sims", nargs="?", type=int, default=30000)
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="9")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lag-ref", type=float)
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""
    if a.tag:
        os.environ["SBI4_LIBRARY"] = os.path.join(MODEL_DIR, f"ramp2_library{tag}.npz")
    if a.lag_ref is not None:
        os.environ["SBI4_LAG_REF"] = str(a.lag_ref)
    from cloudchamber import sbi_shape4 as ss4  # after the environment is set (also seen by the workers)
    from run_sbi_shape import summary, train

    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(ss4.RAMP2_LIBRARY):
        if not a.csv:
            raise SystemExit(f"{ss4.RAMP2_LIBRARY} missing: give reanalysis CSVs (--csv) to build it")
        build_library(a.csv, ss4.RAMP2_LIBRARY, a.start, a.end)
    print(f"library {ss4.RAMP2_LIBRARY}, P_LAG_REF {ss4.P_LAG_REF}", flush=True)
    theta, x = simulate(ss4, a.n_sims, tag)
    n_test = 2000
    th_tr, x_tr, th_te, x_te = theta[:-n_test], x[:-n_test], theta[-n_test:], x[-n_test:]
    post_path = os.path.join(MODEL_DIR, f"npe_shape4{tag}_{a.n_sims}.pt")
    if os.path.exists(post_path):
        posterior = torch.load(post_path, weights_only=False)
    else:
        posterior = train(th_tr, x_tr)
        torch.save(posterior, post_path)
    med, lo, hi = summary(posterior, x_te)
    rmse = np.sqrt(np.mean((med - th_te) ** 2, 0))
    cov = ((th_te >= lo) & (th_te <= hi)).mean(0)
    print(f"held-out simulations: rms qw {rmse[0]:.2f} g/kg, log10 N {rmse[1]:.3f}; "
          f"90% CI coverage qw {100*cov[0]:.0f}%, log10 N {100*cov[1]:.0f}%")
    print(f"saved {post_path}; re-estimate recorded cycles with reanalyze.py START END out.csv {post_path}")


if __name__ == "__main__":
    main()
