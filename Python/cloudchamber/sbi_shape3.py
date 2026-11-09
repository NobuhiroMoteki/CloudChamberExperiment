# -*- coding: utf-8 -*-
"""Tr-shape SBI, version 3: version 2 with the pressure-sensor correction and no wall heating.

Differences from sbi_shape2.py:
  - Corrected moist thermodynamics and state-dependent D_v, k_a, sigma_w (activation-v2).
  - No wall heating: the optically probed core of the flask cools adiabatically.
  - Pressure-sensor correction found from recorded cycles (qw fixed at the hygrometer value, 12
    cycles; validated on 12 others: qw within 6 % of the hygrometer):
      true pressure drop = P_GAIN * measured drop, and the sensor lags the flask by P_LAG,
      i.e. p_flask(t) = p_ini - P_GAIN (p_ini - p_meas(t + P_LAG)).
    Both enter as narrow nuisance ranges. x still holds the measured pressure-fit parameters.
  - Effective optical path fixed at L_PATH3 = 0.18 m.
"""
import numpy as np

from .activation import simulate_activation
from .features import ONSET_TOL
from .sbi_shape import COND_RANGE, THETA_HIGH, THETA_LOW, _blur, sample_cond, sample_theta, shape_features
from .sbi_shape2 import SHAPE_OFFSETS, T_END, OUT_DT, X_NAMES, observed_x

P_GAIN = (0.90, 0.94)  # true / measured pressure drop
P_LAG = (0.25, 0.31)  # [s] lag of the pressure reading behind the flask
L_PATH3 = 0.18  # [m]
__all__ = ["THETA_LOW", "THETA_HIGH", "COND_RANGE", "SHAPE_OFFSETS", "X_NAMES", "observed_x",
           "simulate_x", "simulate_one", "sample_theta", "sample_cond"]


def sample_nuisance(rng):
    return dict(r_g=10 ** rng.uniform(np.log10(20e-9), np.log10(80e-9)), sigma_g=rng.uniform(1.4, 2.2),
                kappa=rng.uniform(0.1, 0.7), alpha_c=10 ** rng.uniform(-1.0, 0.0),
                sigma_rel=10 ** rng.uniform(np.log10(0.015), np.log10(0.10)),
                spread_floor=10 ** rng.uniform(np.log10(0.003), np.log10(0.03)),
                p_gain=rng.uniform(*P_GAIN), p_lag=rng.uniform(*P_LAG),
                blur=rng.uniform(0.0, 0.05), dt0=rng.normal(0.0, 0.01),
                gain=1.0 + rng.normal(0.0, 3e-4), noise=10 ** rng.uniform(-4.5, -3.5))


def simulate_x(theta, cond, nuis, rng, n_bins=12):
    """One noisy observation x (see X_NAMES) for the measured pressure fit `cond`, or None if no
    usable cloud forms."""
    qw, lnN = theta
    c = dict(cond, p_offset=cond["p_ini"] - nuis["p_gain"] * (cond["p_ini"] - cond["p_offset"]))
    s = simulate_activation(qw * 1e-3, 10 ** lnN, r_g=nuis["r_g"], sigma_g=nuis["sigma_g"],
                            kappa=nuis["kappa"], alpha_c=nuis["alpha_c"], n_bins=n_bins,
                            t_end=T_END, out_dt=OUT_DT, sigma_rel=nuis["sigma_rel"], spread_exp=2.0,
                            spread_floor=nuis["spread_floor"], l_path=L_PATH3, **c)
    tr = _blur(s["Tr"], nuis["blur"], OUT_DT)
    # flask time t corresponds to t - p_lag on the clock of the (fitted) pressure reading;
    # t0 estimated with an error dt0
    t = s["t"] - nuis["p_lag"] + nuis["dt0"]
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
