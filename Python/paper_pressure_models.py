# -*- coding: utf-8 -*-
"""Residuals of four pressure models fitted to every recorded expansion (paper Sect. 2.4).

Models, all with p_ini fixed and the onset t0 free, fitted to the whole 10-s expansion phase as in
cloudchamber.features.fit_pressure_ramp2 (Eq. 1 of the paper):
  A  single exponential, instantaneous opening        (f1 = 1, t_open = 0)
  B  single exponential, linear conductance ramp      (f1 = 1)
  C  two exponentials, instantaneous opening          (t_open = 0)
  D  two exponentials, linear conductance ramp        (Eq. 1)
The rms residual [hPa] is reported over the first 2 s after each model's t0 (the cloud forms
within them) and over the whole phase. Cycles: 31 Oct 12:18 to 2 Nov (the evaluation period).

Usage: python paper_pressure_models.py  (also writes data/reanalysis/pressure_models_rms.npz, read by
paper_fig_pressure.py)
"""
import glob
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from cloudchamber.features import despike, find_decompression_start, pressure_ramp2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
START, END = datetime(2026, 10, 31, 12, 18), datetime(2026, 11, 3)
EARLY = 2.0  # [s] window after t0 for the early rms
DP = 200.0  # [Pa] drop that marks the onset, as in features.py
T0_SLACK = 1.0  # [s] t0 may lie this far after the detected onset, so that instantaneous-opening
# models are not handicapped by the bound (features.py bounds t0 at the onset)

# free parameters (t0, p_offset, tau1, f1, tau2, t_open); None = fixed at the value given
MODELS = {
    "A": dict(f1=1.0, t_open=0.0),
    "B": dict(f1=1.0),
    "C": dict(t_open=0.0),
    "D": dict(),
}
NAMES = ("t0", "p_offset", "tau1", "f1", "tau2", "t_open")


def fit(t, p, p0, i, fixed):
    """Least-squares fit of pressure_ramp2 with the parameters in `fixed` held.

    Returns (par, rms_early, rms_all): par in the order of NAMES (t0 on the origin of t) [s, Pa, s, -, s, s],
    rms [Pa].
    """
    x0 = dict(t0=t[i] - 0.25, p_offset=0.4 * p0, tau1=1.9, f1=0.7, tau2=8.0, t_open=0.6)
    lo = dict(t0=t[0], p_offset=0.0, tau1=0.05, f1=0.0, tau2=0.05, t_open=0.0)
    hi = dict(t0=t[i] + T0_SLACK, p_offset=0.99 * p0, tau1=50.0, f1=1.0, tau2=100.0, t_open=3.0)
    scale = dict(t0=0.1, p_offset=1e4, tau1=1.0, f1=0.1, tau2=1.0, t_open=0.1)
    if fixed.get("f1") == 1.0:  # single exponential: tau2 is irrelevant
        fixed = dict(fixed, tau2=8.0)
    free = [n for n in NAMES if n not in fixed]

    def full(q):
        v = dict(fixed, **dict(zip(free, q)))
        return [v[n] for n in NAMES]

    r = least_squares(lambda q: pressure_ramp2(t, p0, *full(q)) - p, [x0[n] for n in free],
                      bounds=([lo[n] for n in free], [hi[n] for n in free]), x_scale=[scale[n] for n in free])
    par = full(r.x)
    res = pressure_ramp2(t, p0, *par) - p
    early = (t >= par[0]) & (t <= par[0] + EARLY)
    return par, float(np.sqrt(np.mean(res[early] ** 2))), float(np.sqrt(np.mean(res ** 2)))


def one(path):
    d = np.load(path)
    t, p0 = d["t"].astype(float), float(d["p_ini"]) * 1e2
    p = despike(d["p"].astype(float) * 1e2)
    i = find_decompression_start(p, p0, DP)
    if i is None or i == 0:
        return None
    try:
        return [fit(t, p, p0, i, fixed)[1:] for fixed in MODELS.values()]
    except ValueError:
        return None


RMS_FILE = os.path.join(DATA, "reanalysis", "pressure_models_rms.npz")


def collect():
    """rms residuals of the four models for every cycle, shape (n_cycles, 4, 2) [hPa] (windows 0-2 s, all)."""
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, "raw_*", "decomp_*.npz"))):
        stamp = datetime.strptime(os.path.basename(f)[7:22], "%Y%m%d_%H%M%S")
        if START <= stamp < END:
            files.append(f)
    with Pool() as pool:
        out = [r for r in pool.map(one, files, chunksize=20) if r is not None]
    a = np.array(out) * 1e-2  # shape: (n_cycles, 4 models, 2 windows) [hPa]
    print(f"{len(files)} files, {len(a)} cycles fitted")
    np.savez(RMS_FILE, rms=a, models=np.array(list(MODELS)))
    return a


def main():
    a = collect()
    print("model  rms 0-2 s [hPa]: median (5-95 %)      rms 0-10 s [hPa]: median (5-95 %)")
    for k, m in enumerate(MODELS):
        e, w = a[:, k, 0], a[:, k, 1]
        print(f"  {m}    {np.median(e):5.2f} ({np.percentile(e, 5):.2f}-{np.percentile(e, 95):.2f})"
              f"            {np.median(w):5.2f} ({np.percentile(w, 5):.2f}-{np.percentile(w, 95):.2f})")


if __name__ == "__main__":
    main()
