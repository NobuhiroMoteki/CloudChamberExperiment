# -*- coding: utf-8 -*-
"""Estimators of (qw, log10 nd) from one analysed cycle.

Interface: estimator(cycle) -> dict(qw_gkg, qw_lo, qw_hi, log10nd, log10nd_lo, log10nd_hi),
where cycle is the dict produced by the controller (features + raw decompression series).
The *_lo / *_hi bounds are a central credible interval (NaN if the estimator has none).
"""
import numpy as np

from .features import zeta_features
from .forward import simulate


class LinearRegressionEstimator:
    """Baseline of Docs Sec. 5.2.2: x = A [1, zeta_moist, zeta_turbid]^T, fitted on a simulated grid."""

    def __init__(self, T_ini=299.86, p_ini=1002e2, p_offset=441.34e2, tau_vac=3.9474,
                 qw_range=(7.0, 16.0), log10nd_range=(9.0, 11.0), ngrid=10):
        X, Y = [], []
        for qw in np.linspace(*qw_range, ngrid):
            for lnd in np.linspace(*log10nd_range, ngrid):
                s = simulate(qw * 1e-3, 10 ** lnd, T_ini=T_ini, p_ini=p_ini,
                             p_offset=p_offset, tau_vac=tau_vac)
                _, _, zm, zt = zeta_features(s["t"], s["p"], s["Tr"])
                if np.isfinite(zm) and np.isfinite(zt):
                    X.append([qw, lnd])
                    Y.append([1.0, zm, zt])
        X, Y = np.array(X).T, np.array(Y).T
        self.A = X @ Y.T @ np.linalg.inv(Y @ Y.T)

    def __call__(self, cycle):
        zm, zt = cycle["zeta_moist"], cycle["zeta_turbid"]
        nan = dict(qw_gkg=np.nan, qw_lo=np.nan, qw_hi=np.nan,
                   log10nd=np.nan, log10nd_lo=np.nan, log10nd_hi=np.nan)
        if not (np.isfinite(zm) and np.isfinite(zt)):
            return nan
        qw, lnd = self.A @ np.array([1.0, zm, zt])
        return dict(nan, qw_gkg=float(qw), log10nd=float(lnd))
