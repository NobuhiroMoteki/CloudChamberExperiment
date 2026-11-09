# -*- coding: utf-8 -*-
"""Simulation-based inference of (qw, log10 N) from the two cycle features (zeta_moist, zeta_turbid).

Used for data that have no Tr(t) waveforms (e.g. the 2021 campaign). The simulator is the activation
model (activation.py); everything that is uncertain but not of interest is drawn at random for each
simulation and therefore marginalised by the trained posterior:
  - microphysics / chamber: condensation coefficient, wall heating, valve opening time
  - analysis conventions of the old LabVIEW program: zeta_turbid as difference or linear-fit slope,
    p_c taken where Tr first reaches a threshold between 0.980 and 0.985
  - feature noise (levels taken from the cycle-to-cycle scatter of the 2021 data)
The chamber state (T_ini, p_ini, p_offset, tau_vac) is measured every cycle and is part of x.
"""
import numpy as np

from .activation import simulate_activation

THETA_LOW = np.array([3.0, 8.0])  # qw [g/kg], log10 N [#/kg]
THETA_HIGH = np.array([22.0, 11.0])
COND_RANGE = {"T_ini": (291.0, 301.0), "p_ini": (990e2, 1020e2),
              "p_offset": (410e2, 480e2), "tau_vac": (3.3, 4.6)}
X_NAMES = ["zeta_moist", "ln_zeta_turbid", "T_ini", "p_ini_hPa", "p_offset_hPa", "tau_vac"]
DT_EVAL = 0.3  # [s]


def sample_theta(rng):
    return rng.uniform(THETA_LOW, THETA_HIGH)


def sample_cond(rng):
    return {k: rng.uniform(*v) for k, v in COND_RANGE.items()}


def sample_nuisance(rng):
    return dict(alpha_c=10 ** rng.uniform(-1.0, 0.0), inv_tau_wall=rng.uniform(0.0, 0.2),
                t_wall_on=rng.uniform(0.0, 1.0), t_open=rng.uniform(0.0, 1.5),
                zeta_diff=rng.random() < 0.5, threshold=rng.uniform(0.980, 0.985),
                sd_zm=rng.uniform(0.001, 0.003), sd_lzt=10 ** rng.uniform(np.log10(0.03), np.log10(0.2)))


def features_from_series(t, p, tr, p_ini, threshold, zeta_diff):
    """(zeta_moist, zeta_turbid) as computed by the LabVIEW program; None if undefined."""
    i = np.nonzero(tr <= threshold)[0]
    if i.size == 0:
        return None
    i = i[0]
    j = i + int(round(DT_EVAL / (t[1] - t[0])))
    if j >= t.size:
        return None
    if zeta_diff:
        zt = (tr[i] - tr[j]) / DT_EVAL
    else:
        zt = -np.polyfit(t[i:j + 1] - t[i], tr[i:j + 1], 1)[0]
    if zt <= 0:
        return None
    return 1.0 - p[i] / p_ini, zt


def simulate_x(theta, cond, nuis, rng, n_bins=8):
    """One noisy observation x (see X_NAMES), or None if no cloud forms within the window."""
    qw, lnN = theta
    s = simulate_activation(qw * 1e-3, 10 ** lnN, n_bins=n_bins, t_end=4.5,
                            alpha_c=nuis["alpha_c"], t_open=nuis["t_open"],
                            tau_wall=1.0 / nuis["inv_tau_wall"] if nuis["inv_tau_wall"] > 0 else np.inf,
                            t_wall_on=nuis["t_wall_on"], **cond)
    f = features_from_series(s["t"], s["p"], s["Tr"], cond["p_ini"], nuis["threshold"], nuis["zeta_diff"])
    if f is None:
        return None
    zm, zt = f
    return np.array([zm + rng.normal(0, nuis["sd_zm"]), np.log(zt) + rng.normal(0, nuis["sd_lzt"]),
                     cond["T_ini"], cond["p_ini"] * 1e-2, cond["p_offset"] * 1e-2, cond["tau_vac"]])


def simulate_one(seed):
    """Draw (theta, x) from prior x simulator; returns None when no cloud forms."""
    rng = np.random.default_rng(seed)
    theta, cond, nuis = sample_theta(rng), sample_cond(rng), sample_nuisance(rng)
    x = simulate_x(theta, cond, nuis, rng)
    return None if x is None else (theta, x)


def observed_x(zeta_moist, zeta_turbid, T_ini, p_ini_hpa, p_offset_hpa, tau_vac):
    """x vector for recorded cycle features (same layout as simulate_x)."""
    return np.array([zeta_moist, np.log(zeta_turbid), T_ini, p_ini_hpa, p_offset_hpa, tau_vac])
