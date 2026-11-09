# -*- coding: utf-8 -*-
"""Tr-shape SBI, version 2: the window extends past the second Mie hump.

Differences from sbi_shape.py (version 1, window t_s-0.10..t_s+0.35 s):
  - Tr is sampled at t_s + SHAPE_OFFSETS = -0.10..+0.90 s (every 0.01 s), which includes the second
    Tr hump (a ~ 3 um, where Qext falls after its broad maximum near 2.8 um; the first hump is at
    a ~ 2 um); its timing constrains wall heating and its depth the liquid water.
  - The droplet-size spread that smooths the Mie structure narrows with condensational growth:
    sigma_rel(a) = max(sigma_1 (0.69 um / a)^2, sigma_floor) (activation.simulate_activation).
  - The effective optical path through the cloud is a nuisance (L_RANGE): the cloud may not fill the
    whole 0.2 m flask diameter.
Wall heating stays a nuisance (it is not assumed negligible).
"""
import numpy as np

from .activation import simulate_activation
from .features import ONSET_TOL
from .sbi_shape import COND_RANGE, THETA_HIGH, THETA_LOW, _blur, sample_cond, sample_theta, shape_features

SHAPE_OFFSETS = np.round(np.arange(-0.10, 0.90 + 1e-9, 0.01), 3)
T_END = 4.2  # [s] simulated time after t0
OUT_DT = 0.005  # [s]
L_RANGE = (0.14, 0.20)  # [m] effective optical path
X_NAMES = (["t_s"] + [f"tr{o:+.2f}" for o in SHAPE_OFFSETS]
           + ["T_ini", "p_ini_hPa", "p_offset_hPa", "tau_vac", "frac1", "tau_vac2", "t_open"])
__all__ = ["THETA_LOW", "THETA_HIGH", "COND_RANGE", "SHAPE_OFFSETS", "X_NAMES", "observed_x",
           "simulate_x", "simulate_one", "sample_theta", "sample_cond"]


def observed_x(t_s, tr_shape, T_ini, p_ini, ramp2):
    """x for a measured cycle; p_ini [Pa], ramp2 = dict from features.fit_pressure_ramp2."""
    return np.r_[t_s, tr_shape, T_ini, p_ini * 1e-2, ramp2["p_offset"] * 1e-2, ramp2["tau_vac"],
                 ramp2["frac1"], ramp2["tau_vac2"], ramp2["t_open"]]


def sample_nuisance(rng):
    return dict(r_g=10 ** rng.uniform(np.log10(20e-9), np.log10(80e-9)), sigma_g=rng.uniform(1.4, 2.2),
                kappa=rng.uniform(0.1, 0.7), alpha_c=10 ** rng.uniform(-1.0, 0.0),
                inv_tau_wall=rng.uniform(0.0, 0.2), t_wall_on=rng.uniform(0.0, 1.0),
                sigma_rel=10 ** rng.uniform(np.log10(0.015), np.log10(0.10)),
                spread_floor=10 ** rng.uniform(np.log10(0.003), np.log10(0.03)),
                l_path=rng.uniform(*L_RANGE),
                blur=rng.uniform(0.0, 0.05), dt0=rng.normal(0.0, 0.01),
                gain=1.0 + rng.normal(0.0, 3e-4), noise=10 ** rng.uniform(-4.5, -3.5))


def simulate_x(theta, cond, nuis, rng, n_bins=12):
    """One noisy observation x (see X_NAMES), or None if no usable cloud forms."""
    qw, lnN = theta
    s = simulate_activation(qw * 1e-3, 10 ** lnN, r_g=nuis["r_g"], sigma_g=nuis["sigma_g"],
                            kappa=nuis["kappa"], alpha_c=nuis["alpha_c"], n_bins=n_bins,
                            tau_wall=1.0 / nuis["inv_tau_wall"] if nuis["inv_tau_wall"] > 0 else np.inf,
                            t_wall_on=nuis["t_wall_on"], t_end=T_END, out_dt=OUT_DT,
                            sigma_rel=nuis["sigma_rel"], spread_exp=2.0, spread_floor=nuis["spread_floor"],
                            l_path=nuis["l_path"], **cond)
    tr = _blur(s["Tr"], nuis["blur"], OUT_DT)
    t = s["t"] + nuis["dt0"]  # t0 estimated with an error dt0
    t_ms = np.arange(0.0, t[-1], 0.01)  # resample like the 100 Hz measurement
    tr_ms = np.interp(t_ms, t, tr) * nuis["gain"] + rng.normal(0.0, nuis["noise"], t_ms.size)
    sd = nuis["noise"]
    f = shape_features(t_ms, tr_ms, onset_level=nuis["gain"] - max(5 * sd, ONSET_TOL), offsets=SHAPE_OFFSETS)
    if f is None:
        return None
    return np.r_[f[0], f[1], cond["T_ini"], cond["p_ini"] * 1e-2, cond["p_offset"] * 1e-2,
                 cond["tau_vac"], cond["frac1"], cond["tau_vac2"], cond["t_open"]]


def simulate_one(seed):
    """Draw (theta, x) from prior x simulator; returns None when no usable cloud forms."""
    rng = np.random.default_rng(seed)
    theta, cond, nuis = sample_theta(rng), sample_cond(rng), sample_nuisance(rng)
    x = simulate_x(theta, cond, nuis, rng)
    return None if x is None else (theta, x)
