# -*- coding: utf-8 -*-
"""Simulation-based inference of (qw, log10 N) from the shape of Tr(t) up to the first Mie hump.

Replaces the zeta_moist / zeta_turbid features for data recorded with Tr(t) waveforms.
Observation x (see X_NAMES):
  - t_s: time [s after the pressure onset t0] at which Tr crosses the onset level
    (baseline - max(5 sd, 0.001), linear interpolation between samples); carries mostly qw
  - Tr at t_s + SHAPE_OFFSETS (-0.10..+0.35 s, every 0.01 s): onset and first hump (droplet radius
    ~ 2 um, where Qext falls after its broad maximum near 1.7 um); carries mostly N.
    Later parts of Tr(t) are not used (wall heat conduction etc. may affect them).
  - chamber state measured every cycle: T_ini, p_ini and the ramp + double-exponential
    pressure parameters (features.fit_pressure_ramp2)
Nuisances drawn per simulation and thereby marginalised: aerosol size distribution and kappa,
condensation coefficient, wall heating, droplet-size spread (smooths the Mie structure),
time blur of Tr (cloud forming at slightly different times along the light path),
error of t0, gain error of I0 and noise of Tr.
"""
import numpy as np

from .activation import simulate_activation
from .features import ONSET_TOL

THETA_LOW = np.array([3.0, 8.0])  # qw [g/kg], log10 N [#/kg]
THETA_HIGH = np.array([20.0, 11.0])
COND_RANGE = {"T_ini": (288.0, 303.0), "p_ini": (980e2, 1030e2), "p_offset": (300e2, 460e2),
              "tau_vac": (1.5, 2.4), "frac1": (0.5, 0.85), "tau_vac2": (4.0, 20.0),
              "t_open": (0.4, 0.85)}
SIGMA_REL_CHOICES = (0.015, 0.03, 0.06, 0.1)
SHAPE_OFFSETS = np.round(np.arange(-0.10, 0.35 + 1e-9, 0.01), 3)
TR_THRESHOLD = 0.985  # a cloud must reach this to be counted
T_END = 3.6  # [s] simulated time after t0
OUT_DT = 0.005  # [s]
X_NAMES = (["t_s"] + [f"tr{o:+.2f}" for o in SHAPE_OFFSETS]
           + ["T_ini", "p_ini_hPa", "p_offset_hPa", "tau_vac", "frac1", "tau_vac2", "t_open"])


def onset_time(t, tr, onset_level=1.0 - ONSET_TOL, tr_threshold=TR_THRESHOLD):
    """Sub-sample time where Tr last crosses onset_level before first reaching tr_threshold."""
    below = np.nonzero(tr <= tr_threshold)[0]
    if below.size == 0:
        return np.nan
    ic = below[0]
    above = np.nonzero(tr[:ic] >= onset_level)[0]
    if above.size == 0:
        return np.nan
    i = above[-1]
    f = (tr[i] - onset_level) / (tr[i] - tr[i + 1])  # tr[i+1] < onset_level <= tr[i]
    return t[i] + f * (t[i + 1] - t[i])


def shape_features(t, tr, onset_level=1.0 - ONSET_TOL, offsets=SHAPE_OFFSETS):
    """(t_s, Tr on t_s + offsets) from t [s after t0] and Tr; None if undefined."""
    ok = np.isfinite(tr)
    t, tr = t[ok], tr[ok]
    t_s = onset_time(t, tr, onset_level)
    if not np.isfinite(t_s) or t_s + offsets[-1] > t[-1] or t_s + offsets[0] < t[0]:
        return None
    return t_s, np.interp(t_s + offsets, t, tr)


def observed_x(t_s, tr_shape, T_ini, p_ini, ramp2):
    """x for a measured cycle; p_ini [Pa], ramp2 = dict from features.fit_pressure_ramp2."""
    return np.r_[t_s, tr_shape, T_ini, p_ini * 1e-2, ramp2["p_offset"] * 1e-2, ramp2["tau_vac"],
                 ramp2["frac1"], ramp2["tau_vac2"], ramp2["t_open"]]


def sample_theta(rng):
    return rng.uniform(THETA_LOW, THETA_HIGH)


def sample_cond(rng):
    return {k: rng.uniform(*v) for k, v in COND_RANGE.items()}


def sample_nuisance(rng):
    return dict(r_g=10 ** rng.uniform(np.log10(20e-9), np.log10(80e-9)), sigma_g=rng.uniform(1.4, 2.2),
                kappa=rng.uniform(0.1, 0.7), alpha_c=10 ** rng.uniform(-1.0, 0.0),
                inv_tau_wall=rng.uniform(0.0, 0.2), t_wall_on=rng.uniform(0.0, 1.0),
                sigma_rel=SIGMA_REL_CHOICES[rng.integers(len(SIGMA_REL_CHOICES))],
                blur=rng.uniform(0.0, 0.05), dt0=rng.normal(0.0, 0.01),
                gain=1.0 + rng.normal(0.0, 3e-4), noise=10 ** rng.uniform(-4.5, -3.5))


def _blur(tr, sd, dt):
    if sd < dt:
        return tr
    k = np.arange(-int(4 * sd / dt), int(4 * sd / dt) + 1) * dt
    w = np.exp(-0.5 * (k / sd) ** 2)
    w /= w.sum()
    pad = np.r_[np.full(k.size // 2, tr[0]), tr, np.full(k.size // 2, tr[-1])]
    return np.convolve(pad, w, mode="valid")


def simulate_x(theta, cond, nuis, rng, n_bins=12):
    """One noisy observation x (see X_NAMES), or None if no usable cloud forms."""
    qw, lnN = theta
    s = simulate_activation(qw * 1e-3, 10 ** lnN, r_g=nuis["r_g"], sigma_g=nuis["sigma_g"],
                            kappa=nuis["kappa"], alpha_c=nuis["alpha_c"], n_bins=n_bins,
                            tau_wall=1.0 / nuis["inv_tau_wall"] if nuis["inv_tau_wall"] > 0 else np.inf,
                            t_wall_on=nuis["t_wall_on"], t_end=T_END, out_dt=OUT_DT,
                            sigma_rel=nuis["sigma_rel"], **cond)
    tr = _blur(s["Tr"], nuis["blur"], OUT_DT)
    t = s["t"] + nuis["dt0"]  # t0 estimated with an error dt0
    t_ms = np.arange(0.0, t[-1], 0.01)  # resample like the 100 Hz measurement
    tr_ms = np.interp(t_ms, t, tr) * nuis["gain"] + rng.normal(0.0, nuis["noise"], t_ms.size)
    sd = nuis["noise"]
    f = shape_features(t_ms, tr_ms, onset_level=nuis["gain"] - max(5 * sd, ONSET_TOL))
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
