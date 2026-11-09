# -*- coding: utf-8 -*-
"""Analysis of one decompression phase: pressure fit, cloud onset and features."""
import numpy as np

from .features import ONSET_TOL, fit_pressure, turbid_onset, zeta_features

# baseline check on Tr before the pressure starts to fall (not just before t_c: at low N, Tr starts
# to fall slowly and a window before t_c flagged most good cycles)
BASELINE_PERCENTILE = 20
BASELINE_RANGE = (0.995, 1.005)
ONSET_NSIGMA = 5.0  # onset level = baseline - max(ONSET_NSIGMA * noise sd, ONSET_TOL)


def analyze_decompression(t, p, tr, p_ini, tr_threshold=0.985, dt_eval=0.3, fit_window=3.0):
    """t [s] from the start of the decompression phase, p [Pa], tr [-], p_ini [Pa].

    Returns a dict; times t0 and t_c are relative to the phase start and to t0, respectively.
    Flags: 'no_onset' (pressure did not fall), 'no_cloud', 'unstable_baseline', 'nan_I0'.
    """
    res = dict(t0=np.nan, p_offset=np.nan, tau_vac=np.nan, t_c=np.nan, t_s=np.nan, p_c=np.nan,
               zeta_moist=np.nan, zeta_turbid=np.nan, flags=[])
    if not np.isfinite(tr).any():
        res["flags"].append("nan_I0")
    t0, p_offset, tau = fit_pressure(t, p, p_ini, fit_window=fit_window)
    if not np.isfinite(t0):
        res["flags"].append("no_onset")
        return res
    res.update(t0=t0, p_offset=p_offset, tau_vac=tau)

    # Tr baseline and its noise from the samples before the pressure starts to fall
    pre = tr[(t < t0) & np.isfinite(tr)]
    if pre.size >= 10:
        onset_level = np.median(pre) - max(ONSET_NSIGMA * np.std(pre), ONSET_TOL)
    else:
        onset_level = 1.0 - ONSET_TOL
    if pre.size >= 2 and not BASELINE_RANGE[0] <= np.percentile(pre, BASELINE_PERCENTILE) <= BASELINE_RANGE[1]:
        res["flags"].append("unstable_baseline")
    sel = t >= t0
    t_c, p_c, zm, zt = zeta_features(t[sel] - t0, p[sel], tr[sel], p0=p_ini,
                                     tr_threshold=tr_threshold, dt_eval=dt_eval,
                                     onset_level=onset_level)
    res.update(t_c=t_c, p_c=p_c, zeta_moist=zm, zeta_turbid=zt)
    if not np.isfinite(t_c):
        if "nan_I0" not in res["flags"]:
            res["flags"].append("no_cloud")
        return res
    ts, trs = t[sel] - t0, tr[sel]
    res["t_s"] = ts[turbid_onset(trs, np.searchsorted(ts, t_c), onset_level)]
    return res
