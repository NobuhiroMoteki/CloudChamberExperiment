# -*- coding: utf-8 -*-
"""Sensitivity of the estimates to the thermal accommodation coefficient alpha_t (paper Sect. 3.2).

Tr(t) simulated with alpha_t = 0.85 (lower bound of Winkler et al. 2004) is taken as the truth, and the
model with alpha_t = 1 (as used) is fitted to it over the inference window (rms of Tr): N alone, and
qw and N together. Nominal aerosol (r_g 40 nm, sigma_g 1.8, kappa 0.3), 12 bins, typical pressure
curve; qw 5.8-9.2 g/kg, N 500-8000 cm-3, alpha_c 0.1, 0.3, 1.
Printed per case: max |dTr| in the window, dN% with N alone, dN% and dqw% from the joint fit.

Usage: python paper_alpha_t_sensitivity.py   (about 2 min)
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloudchamber.activation import simulate_activation
from cloudchamber.sbi_shape import _blur
from cloudchamber.sbi_shape2 import OUT_DT, T_END
from cloudchamber.sbi_shape3 import L_PATH3
PRES = dict(T_ini=296.0, p_ini=985e2, p_offset=440e2, tau_vac=2.5, frac1=0.7, tau_vac2=8.0, t_open=0.6)
RHO_A = 985e2 / (287.05 * 296.0)
def run(qw, n_cm3, alpha_t, alpha_c=0.3):
    s = simulate_activation(qw, n_cm3*1e6/RHO_A, r_g=40e-9, sigma_g=1.8, kappa=0.3, alpha_c=alpha_c,
                            alpha_t=alpha_t, n_bins=12, t_end=T_END, out_dt=OUT_DT, sigma_rel=0.04,
                            spread_exp=2.0, spread_floor=0.01, l_path=L_PATH3, forward=True, **PRES)
    return s["t"], _blur(s["Tr"], 0.02, OUT_DT)
print(f"{'qw':>5s} {'N':>6s} {'alpha_c':>7s} {'max|dTr|':>9s} {'dN% (N only)':>13s} {'dN% joint':>10s} {'dqw% joint':>11s}")
for alpha_c in (0.3, 1.0, 0.1):
  for qw, n in ((5.8e-3, 2000), (7e-3, 500), (7e-3, 2000), (7e-3, 8000), (9.2e-3, 2000)):
    t, tr1 = run(qw, n, 1.0, alpha_c); _, tr85 = run(qw, n, 0.85, alpha_c)
    i = np.argmax(tr85 < 0.995); m = (t >= t[i]-0.1) & (t <= t[i]+0.9)
    d = np.max(np.abs(tr1[m]-tr85[m]))
    # model with alpha_t = 1 fitted to "truth" alpha_t = 0.85
    c1 = lambda lg: np.sqrt(np.mean((run(qw, 10**lg, 1.0, alpha_c)[1][m]-tr85[m])**2))
    r1 = minimize_scalar(c1, bounds=(np.log10(n)-0.1, np.log10(n)+0.1), method="bounded", options=dict(xatol=1e-4))
    c2 = lambda x: np.sqrt(np.mean((run(qw*(1+x[1]), 10**x[0], 1.0, alpha_c)[1][m]-tr85[m])**2))
    r2 = minimize(c2, [np.log10(n), 0.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7))
    print(f"{qw*1e3:5.1f} {n:6d} {alpha_c:7.1f} {d:9.5f} {(10**r1.x/n-1)*100:13.2f} {(10**r2.x[0]/n-1)*100:10.2f} {r2.x[1]*100:11.2f}", flush=True)
