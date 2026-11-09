# -*- coding: utf-8 -*-
"""Decompression-phase analysis shared by measured and simulated data.

Definitions (agreed with the LabVIEW version, Docs Sec. 4.5 and 5.2.1):
- t = 0 is when the chamber pressure actually starts to fall (not when V2 is commanded open).
  The pressure at that instant equals the plateau before it, so p0 = p_ini (median of the last
  1 s before decompression), and the onset time t0 is a fit parameter (sub-sample resolution).
- p(t) = (p0 - p_offset)*exp(-t/tau_vac) + p_offset is fitted over t = 0..3 s.
- t_c is the first sample with Tr <= tr_threshold; p_c = p(t_c); zeta_moist = 1 - p_c/p0.
- t_s is where Tr starts to fall from 1: the last sample before t_c with Tr >= onset_level
  (baseline minus a noise margin, see analysis.py).
- zeta_turbid = -(slope of a straight-line fit of Tr over t_s..t_s+0.3 s) [s-1].
"""
import numpy as np
from scipy.optimize import curve_fit, least_squares

ONSET_TOL = 1e-3  # minimum depth below the baseline that counts as "Tr has started to fall"
ONSET_RUN = 3  # consecutive samples below p_ini - dp needed, so a single-sample spike is not the onset


def despike(p):
    """3-point running median (end samples kept): removes the single-sample spikes that the valve
    motor puts on the pressure input and leaves a monotonic curve unchanged."""
    p = np.asarray(p, dtype=float)
    if p.size < 3:
        return p.copy()
    q = p.copy()
    q[1:-1] = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)
    return q


def find_decompression_start(p, p_ini, dp=200.0, run=ONSET_RUN):
    """Index of the first of `run` consecutive samples with p < p_ini - dp [Pa], or None if the
    pressure never falls."""
    below = np.asarray(p < p_ini - dp, dtype=int)
    if below.size < run:
        return None
    idx = np.nonzero(np.convolve(below, np.ones(run, dtype=int), "valid") == run)[0]
    return int(idx[0]) if idx.size else None


def _p_model(t, p_offset, tau_vac, t0, p0):
    dt = np.maximum(t - t0, 0.0)
    return (p0 - p_offset) * np.exp(-dt / tau_vac) + p_offset


def fit_pressure(t, p, p0, fit_window=3.0, p_offset_guess=None, tau_guess=4.0, dp=200.0):
    """Fit Eq. (111) with p0 fixed and the onset time t0 free.

    t, p: samples of the decompression phase (any time origin), p0: plateau pressure [Pa].
    Returns (t0 [s, same origin as t], p_offset [Pa], tau_vac [s]); NaNs if the fit fails.
    """
    p = despike(p)
    i = find_decompression_start(p, p0, dp)
    if i is None or i == 0:
        return np.nan, np.nan, np.nan
    if p_offset_guess is None:
        p_offset_guess = 0.45 * p0
    # first guess of t0: back-extrapolate the initial slope (p0 - p_offset)/tau to p = p0
    t0_guess = t[i] - dp * tau_guess / (p0 - p_offset_guess)
    sel = (t >= t0_guess - 0.5) & (t <= t0_guess + fit_window)
    try:
        popt, _ = curve_fit(lambda tt, po, tau, t0: _p_model(tt, po, tau, t0, p0), t[sel], p[sel],
                            p0=(p_offset_guess, tau_guess, t0_guess),
                            bounds=((0.0, 0.05, t[0]), (p0, 100.0, t[i])))
        # refit on exactly t0..t0+fit_window
        sel = (t >= popt[2] - 0.5) & (t <= popt[2] + fit_window)
        popt, _ = curve_fit(lambda tt, po, tau, t0: _p_model(tt, po, tau, t0, p0), t[sel], p[sel],
                            p0=popt, bounds=((0.0, 0.05, t[0]), (p0, 100.0, t[i])))
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    return float(popt[2]), float(popt[0]), float(popt[1])


def pressure_ramp2(t, p0, t0, p_offset, tau_vac, frac1, tau_vac2, t_open):
    """Valve-ramp + double-exponential pressure model (same as activation.py), t on any origin."""
    x = np.maximum(np.asarray(t, dtype=float) - t0, 0.0)
    if t_open > 0:
        G = np.where(x < t_open, 0.5 * x * x / t_open, x - 0.5 * t_open)
    else:
        G = x
    return p_offset + (p0 - p_offset) * (frac1 * np.exp(-G / tau_vac)
                                         + (1.0 - frac1) * np.exp(-G / tau_vac2))


RAMP2_NAMES = ("t0", "p_offset", "tau_vac", "frac1", "tau_vac2", "t_open")


def fit_pressure_ramp2(t, p, p0, dp=200.0):
    """Fit pressure_ramp2 (p0 fixed) to the whole decompression phase.

    The valve ramp and the second exponential are both needed to reproduce the first ~2 s
    (rms 0.9 hPa vs 4 hPa for Eq. 111). p_offset and tau_vac2 are strongly correlated (only
    their combination is determined), which does not matter for reproducing p(t).
    Returns a dict with RAMP2_NAMES (t0 on the origin of t) and 'p_rms' [Pa]; NaNs on failure.
    """
    nan = dict.fromkeys(RAMP2_NAMES + ("p_rms",), np.nan)
    p = despike(p)
    i = find_decompression_start(p, p0, dp)
    if i is None or i == 0:
        return nan
    x0 = [t[i] - 0.25, 0.4 * p0, 1.9, 0.7, 8.0, 0.6]
    lo = [t[0], 0.0, 0.05, 0.0, 0.05, 0.0]
    hi = [t[i], 0.99 * p0, 50.0, 1.0, 100.0, 3.0]
    try:
        r = least_squares(lambda q: pressure_ramp2(t, p0, *q) - p, x0, bounds=(lo, hi),
                          x_scale=[0.1, 1e4, 1.0, 0.1, 1.0, 0.1])
    except ValueError:
        return nan
    out = dict(zip(RAMP2_NAMES, (float(v) for v in r.x)))
    out["p_rms"] = float(np.sqrt(np.mean(r.fun ** 2)))
    return out


def turbid_onset(tr, ic, onset_level=1.0 - ONSET_TOL):
    """Index t_s where Tr starts to fall: last sample before ic with Tr >= onset_level (0 if none)."""
    above = np.nonzero(tr[:ic] >= onset_level)[0]
    return int(above[-1]) if above.size else 0


def zeta_features(t, p, tr, p0=None, tr_threshold=0.985, dt_eval=0.3, onset_level=1.0 - ONSET_TOL):
    """Return (t_c, p_c, zeta_moist, zeta_turbid); NaNs where undefined (no cloud, window cut off)."""
    if p0 is None:
        p0 = p[0]
    below = np.nonzero(tr <= tr_threshold)[0]
    if below.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    ic = below[0]
    t_c, p_c = t[ic], p[ic]
    zeta_moist = 1.0 - p_c / p0
    t_s = t[turbid_onset(tr, ic, onset_level)]
    if t[-1] < t_s + dt_eval - 1e-9:
        return t_c, p_c, zeta_moist, np.nan
    win = (t >= t_s - 1e-9) & (t <= t_s + dt_eval + 1e-9)
    slope = np.polyfit(t[win] - t_s, tr[win], 1)[0]
    return t_c, p_c, zeta_moist, -slope
