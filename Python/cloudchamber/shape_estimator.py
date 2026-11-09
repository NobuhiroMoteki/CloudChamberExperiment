# -*- coding: utf-8 -*-
"""Tr-shape SBI estimator for recorded cycles.

The newest trained version is used:
  4 (sbi_shape4.py / run_sbi_shape4.py, models/npe_shape4_*.pt): version 3 with the pressure curve
    given by its drop at fixed times and the re-fitted pressure-sensor correction;
  3 (sbi_shape3.py / run_sbi_shape3.py, models/npe_shape3_*.pt): version 2 with corrected
    thermodynamics, the pressure-sensor correction, no wall heating and L = 0.18 m;
  2 (sbi_shape2.py / run_sbi_shape2.py, models/npe_shape2_*.pt): Tr up to the second hump,
    t_s + 0.90 s;
  1 (sbi_shape.py / run_sbi_shape.py, models/npe_shape_*.pt): up to the first hump, t_s + 0.35 s.
Versions 2-4 share the Tr window (SHAPE_OFFSETS); version 4 has a different pressure part of x.
"""
import glob
import os
import re

import numpy as np

from . import sbi_shape as ss
from . import sbi_shape2 as ss2
from . import sbi_shape4 as ss4
from .features import ONSET_TOL, fit_pressure_ramp2

ONSET_NSIGMA = 5.0
MAX_SAMPLING_S = 20.0
MODEL_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "models")
# version -> (file prefix, Tr sampling offsets after t_s, name written to the CSV 'estimator' column,
#             x builder)
VERSIONS = {4: ("npe_shape4_", ss2.SHAPE_OFFSETS, "ShapeSBIEstimator4", ss4.observed_x),
            3: ("npe_shape3_", ss2.SHAPE_OFFSETS, "ShapeSBIEstimator3", ss.observed_x),
            2: ("npe_shape2_", ss2.SHAPE_OFFSETS, "ShapeSBIEstimator2", ss.observed_x),
            1: ("npe_shape_", ss.SHAPE_OFFSETS, "ShapeSBIEstimator", ss.observed_x)}


def cycle_x(t, p, tr, p_ini, T_ini, offsets=ss.SHAPE_OFFSETS, observed_x=ss.observed_x):
    """x for one decompression phase: t [s] from the phase start, p [Pa], Tr, p_ini [Pa], T_ini [K];
    offsets: times after t_s at which Tr is sampled, observed_x: x builder (both model dependent).

    Returns (x or None, info dict with the pressure fit and the reason if None).
    """
    ramp2 = fit_pressure_ramp2(t, p, p_ini)
    if not np.isfinite(ramp2["t0"]):
        return None, dict(ramp2, reason="no_onset")
    pre = tr[(t < ramp2["t0"]) & np.isfinite(tr)]
    if pre.size >= 10:
        level = np.median(pre) - max(ONSET_NSIGMA * np.std(pre), ONSET_TOL)
    else:
        level = 1.0 - ONSET_TOL
    f = ss.shape_features(t - ramp2["t0"], tr, onset_level=level, offsets=offsets)
    if f is None:
        return None, dict(ramp2, reason="no_cloud")
    return observed_x(f[0], f[1], T_ini, p_ini, ramp2), dict(ramp2, t_s=f[0])


def model_version(path):
    """1-4 from the file name (npe_shape_N.pt / npe_shape2_N.pt / ... / npe_shape4_N.pt)."""
    name = os.path.basename(path)
    return next((v for v in (4, 3, 2) if name.startswith(VERSIONS[v][0])), 1)


def latest_model(version=None):
    """Path of the trained posterior of the newest available version (or of the given version)
    with the most simulations, or None."""
    for v in ([version] if version else sorted(VERSIONS, reverse=True)):
        prefix = VERSIONS[v][0]
        paths = [q for q in glob.glob(os.path.join(MODEL_DIR, prefix + "*.pt"))
                 if re.fullmatch(re.escape(prefix) + r"\d+\.pt", os.path.basename(q))]
        if paths:
            return max(paths, key=lambda q: int(os.path.basename(q)[len(prefix):-3]))
    return None


class ShapeSBIEstimator:
    """estimator(cycle) -> dict(qw_gkg, qw_lo, qw_hi, log10nd, log10nd_lo, log10nd_hi, ...)
    with the posterior median and the central 90% credible interval."""

    def __init__(self, path=None, n_samples=1000, version=None):
        import torch
        self._torch = torch
        self.path = path or latest_model(version)
        if self.path is None:
            raise FileNotFoundError("no models/npe_shape*_*.pt; run run_sbi_shape2.py first")
        self.version = model_version(self.path)
        _, self.offsets, self.name, self.observed_x = VERSIONS[self.version]
        self.posterior = torch.load(self.path, weights_only=False)
        self.n = n_samples

    def __call__(self, cycle):
        nan = dict(qw_gkg=np.nan, qw_lo=np.nan, qw_hi=np.nan,
                   log10nd=np.nan, log10nd_lo=np.nan, log10nd_hi=np.nan)
        x, info = cycle_x(cycle["t"], cycle["p"], cycle["tr"], cycle["p_ini"], cycle["T_ini"],
                          offsets=self.offsets, observed_x=self.observed_x)
        extra = {k: info.get(k, np.nan) for k in ("frac1", "tau_vac2", "t_open", "p_rms")}
        extra["t_s_shape"] = info.get("t_s", np.nan)  # after t0_ramp2 (analysis.t_s is after t0)
        extra["t0_ramp2"], extra["p_offset2"], extra["tau_vac1"] = (info.get("t0", np.nan),
                                                                     info.get("p_offset", np.nan),
                                                                     info.get("tau_vac", np.nan))
        # why no estimate was made; the controller appends it to the cycle's flags
        if x is None:
            return dict(nan, **extra, sbi_reason="sbi_" + info["reason"])
        if not np.all(np.isfinite(x)):
            return dict(nan, **extra, sbi_reason="sbi_x_nan")
        # x far outside the training set makes the rejection of out-of-prior samples nearly endless;
        # cap it well below the 60 s cycle and give NaN if too few samples were accepted
        s = self.posterior.sample((self.n,), x=self._torch.tensor(x, dtype=self._torch.float32),
                                  show_progress_bars=False, max_sampling_time=MAX_SAMPLING_S,
                                  return_partial_on_timeout=True).numpy()
        if len(s) < self.n // 2:
            return dict(nan, **extra, sbi_reason="sbi_timeout")
        med, lo, hi = np.median(s, 0), np.percentile(s, 5, 0), np.percentile(s, 95, 0)
        return dict(qw_gkg=float(med[0]), qw_lo=float(lo[0]), qw_hi=float(hi[0]),
                    log10nd=float(med[1]), log10nd_lo=float(lo[1]), log10nd_hi=float(hi[1]), **extra)
