# -*- coding: utf-8 -*-
"""Tr-shape SBI, version 4: version 3 with the pressure curve given by its values at fixed times.

Differences from sbi_shape3.py:
  - The parameters of the pressure fit (features.fit_pressure_ramp2) are degenerate: p_offset,
    tau_vac2 and frac1 trade off against each other (p_offset 0-420 hPa, tau_vac2 3-90 s for curves
    that agree within 2 % over 0-4 s), so ~60 % of the recorded cycles fell outside the training
    ranges. x now holds the measured pressure drop p_ini - p(t0 + PRESS_TIMES) of the fitted curve,
    which is unique.
  - Pressure curves are drawn from a library of fits to recorded cycles (RAMP2_LIBRARY, built by
    run_sbi_shape4.py), widened by random scalings of the drop (DROP_SCALE) and of time
    (TIME_SCALE) and a shift of the valve-opening time (T_OPEN_SHIFT).
  - The photodiode (S1223-01, 3.6 x 3.6 mm at the flask wall; 3 x 3 mm was assumed before 2026-11-03)
    also collects the light scattered forward into the cone it subtends; Qext is replaced by the
    effective Qext (mie.qext_forward; factor 0.93 at a = 2 um, 0.89 at 3 um). Without it N comes out
    ~30 % below the CPC.
  - Pressure-sensor correction re-fitted on 400 cycles from 2026-10-31 12:18 to 2026-11-02 16:48 with
    qw fixed at the hygrometer value (fit_pressure_correction.py with the effective Qext; hygrometer
    qw 5.8-9.2 g/kg). Gain and lag are degenerate along a valley,
    lag = P_LAG_REF + LAG_SLOPE (gain - P_GAIN_REF); at gain 0.96 the best lag is 0.05 / 0.06 / 0.09 s
    for alpha_c 1 / 0.3 / 0.1, so (gain, lag) are drawn along the valley. The v3 ranges
    (gain 0.90-0.94, lag 0.25-0.31 s) lie off the valley.
  - Since 2026-11-08 the correction is off by default (gain 1, lag 0; the paper's models). It used the
    hygrometer as a reference, and leave-one-day-out models without it agree better with both the
    hygrometer and the CPC. SBI4_PCORR=1 restores the valley prior above.
"""
import os

import numpy as np

from .activation import simulate_activation
from .features import ONSET_TOL, pressure_ramp2
from .sbi_shape import COND_RANGE, THETA_HIGH, THETA_LOW, _blur, sample_theta, shape_features
from .sbi_shape2 import OUT_DT, SHAPE_OFFSETS, T_END
from .sbi_shape3 import L_PATH3

# Variants (e.g. calibrated on one period only, run_sbi_shape4.py --tag) are set through environment
# variables so that the simulation worker processes see them too.
P_GAIN = (0.92, 1.00)  # true / measured pressure drop (re-fitted, see module docstring)
P_GAIN_REF, LAG_SLOPE = 0.96, -1.33  # [-], [s]
P_LAG_REF = float(os.environ.get("SBI4_LAG_REF", 0.07))  # [s] lag of the reading on the valley at P_GAIN_REF
LAG_HALF = 0.03  # [s] half width around the valley
PRESS_TIMES = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # [s] after t0
DROP_SCALE = (0.95, 1.05)
TIME_SCALE = (0.95, 1.05)
T_OPEN_SHIFT = (-0.05, 0.05)  # [s]
RAMP2_LIBRARY = os.environ.get("SBI4_LIBRARY") or os.path.join(os.path.dirname(__file__), os.pardir, "models",
                                                               "ramp2_library.npz")
# variants: plain Qext (SBI4_FORWARD=0), pressure-sensor correction on (SBI4_PCORR=1)
FORWARD = os.environ.get("SBI4_FORWARD", "1") == "1"
PCORR = os.environ.get("SBI4_PCORR", "0") == "1"
# sensor response of single forward runs (paper figures, fits): none unless SBI4_PCORR=1
P_GAIN_NOM = P_GAIN_REF if PCORR else 1.0  # [-]
P_LAG_NOM = P_LAG_REF if PCORR else 0.0  # [s]
DT0_SD = 0.01  # [s] s.d. of the error of the valve-opening onset t0 (prior of dt0)
# [s] half width of the time shift scanned in fits: the lag prior with the correction, else 2 s.d. of dt0
LAG_HALF_NOM = float(os.environ.get("SBI4_FIT_LAG_HALF", LAG_HALF if PCORR else 2 * DT0_SD))
LIB_KEYS = ("p_offset_rel", "tau_vac", "frac1", "tau_vac2", "t_open")
X_NAMES = (["t_s"] + [f"tr{o:+.2f}" for o in SHAPE_OFFSETS] + ["T_ini", "p_ini_hPa"]
           + [f"dp{t:.2f}_hPa" for t in PRESS_TIMES])
__all__ = ["THETA_LOW", "THETA_HIGH", "SHAPE_OFFSETS", "X_NAMES", "PRESS_TIMES", "observed_x",
           "simulate_x", "simulate_one", "sample_theta", "sample_cond"]
_LIB = {}


def pressure_drops(p_ini, ramp2):
    """p_ini - p(t0 + PRESS_TIMES) [hPa] of a ramp2 fit (p_ini and ramp2['p_offset'] in Pa)."""
    p = pressure_ramp2(PRESS_TIMES, p_ini, 0.0, ramp2["p_offset"], ramp2["tau_vac"], ramp2["frac1"],
                       ramp2["tau_vac2"], ramp2["t_open"])
    return (p_ini - p) * 1e-2


def observed_x(t_s, tr_shape, T_ini, p_ini, ramp2):
    """x for a measured cycle; p_ini [Pa], ramp2 = dict from features.fit_pressure_ramp2."""
    return np.r_[t_s, tr_shape, T_ini, p_ini * 1e-2, pressure_drops(p_ini, ramp2)]


def library():
    if not _LIB:
        d = np.load(RAMP2_LIBRARY)
        _LIB.update({k: d[k] for k in LIB_KEYS})
    return _LIB


def sample_cond(rng):
    """Chamber state: T_ini, p_ini from COND_RANGE and a widened recorded pressure curve."""
    lib = library()
    i = rng.integers(lib["tau_vac"].size)
    p_ini = rng.uniform(*COND_RANGE["p_ini"])
    a, s = rng.uniform(*DROP_SCALE), rng.uniform(*TIME_SCALE)  # s > 1: faster
    return dict(T_ini=rng.uniform(*COND_RANGE["T_ini"]), p_ini=p_ini,
                p_offset=p_ini * (1.0 - a * (1.0 - lib["p_offset_rel"][i])),
                tau_vac=lib["tau_vac"][i] / s, frac1=lib["frac1"][i], tau_vac2=lib["tau_vac2"][i] / s,
                t_open=max(lib["t_open"][i] / s + rng.uniform(*T_OPEN_SHIFT), 0.05))


def sample_nuisance(rng):
    p_gain = rng.uniform(*P_GAIN)
    nuis = dict(r_g=10 ** rng.uniform(np.log10(20e-9), np.log10(80e-9)), sigma_g=rng.uniform(1.4, 2.2),
                kappa=rng.uniform(0.1, 0.7), alpha_c=10 ** rng.uniform(-1.0, 0.0),
                sigma_rel=10 ** rng.uniform(np.log10(0.015), np.log10(0.10)),
                spread_floor=10 ** rng.uniform(np.log10(0.003), np.log10(0.03)),
                p_gain=p_gain,
                p_lag=P_LAG_REF + LAG_SLOPE * (p_gain - P_GAIN_REF) + rng.uniform(-LAG_HALF, LAG_HALF),
                blur=rng.uniform(0.0, 0.05), dt0=rng.normal(0.0, DT0_SD),
                gain=1.0 + rng.normal(0.0, 3e-4), noise=10 ** rng.uniform(-4.5, -3.5))
    if not PCORR:
        nuis.update(p_gain=1.0, p_lag=0.0)
    return nuis


def simulate_x(theta, cond, nuis, rng, n_bins=12):
    """One noisy observation x (see X_NAMES) for the measured pressure curve `cond`, or None if no
    usable cloud forms."""
    qw, lnN = theta
    c = dict(cond, p_offset=cond["p_ini"] - nuis["p_gain"] * (cond["p_ini"] - cond["p_offset"]))
    s = simulate_activation(qw * 1e-3, 10 ** lnN, r_g=nuis["r_g"], sigma_g=nuis["sigma_g"],
                            kappa=nuis["kappa"], alpha_c=nuis["alpha_c"], n_bins=n_bins,
                            t_end=T_END, out_dt=OUT_DT, sigma_rel=nuis["sigma_rel"], spread_exp=2.0,
                            spread_floor=nuis["spread_floor"], l_path=L_PATH3, forward=FORWARD, **c)
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
    return observed_x(f[0], f[1], cond["T_ini"], cond["p_ini"], cond)


def simulate_one(seed):
    """Draw (theta, x) from prior x simulator; returns None when no usable cloud forms."""
    rng = np.random.default_rng(seed)
    theta, cond, nuis = sample_theta(rng), sample_cond(rng), sample_nuisance(rng)
    x = simulate_x(theta, cond, nuis, rng)
    return None if x is None else (theta, x)
