# -*- coding: utf-8 -*-
"""Discretisation test of the dry-size distribution in the forward model (SI Sect. S2).

Tr(t) with 12 equal-number bins (as used) is compared with a reference of 96 equal-number bins and
with ~60 log-spaced bins whose number follows the lognormal (emulated by repeating radii so that
the unmodified _simulate, which gives every entry the same number, can be used).
For each case: max |dTr| in the inference window and the N that the 12-bin model needs to match
the reference Tr (rms over the window), as a relative change ("dN12%"; "dNlog%" is the same against
the log-spaced Tr). Also printed: peak supersaturation and activated fractions (12 bins, reference).
The model code is not changed: dry_radii is replaced temporarily to inject the radii.

Usage: python paper_bin_convergence.py   (prints one line per case; about 20 s in total)
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloudchamber import activation as act
from cloudchamber.sbi_shape import _blur
from cloudchamber.sbi_shape2 import OUT_DT, T_END
from cloudchamber.sbi_shape3 import L_PATH3

PRES = dict(T_ini=296.0, p_ini=985e2, p_offset=440e2, tau_vac=2.5, frac1=0.7, tau_vac2=8.0, t_open=0.6)
RHO_A = 985e2 / (287.05 * 296.0)  # [kg m-3] to convert cm-3 to kg-1


def run(qw, n_cm3, ad, nom):
    n_kg = n_cm3 * 1e6 / RHO_A
    orig = act.dry_radii
    act.dry_radii = lambda r_g, s_g, nb: ad  # inject the radii; weights equal per entry
    try:
        s = act.simulate_activation(qw, n_kg, r_g=nom["r_g"], sigma_g=nom["sigma_g"], kappa=nom["kappa"],
                                    alpha_c=nom["alpha_c"], n_bins=len(ad), t_end=T_END, out_dt=OUT_DT,
                                    sigma_rel=0.04, spread_exp=2.0, spread_floor=0.01, l_path=L_PATH3,
                                    forward=True, **PRES)
    finally:
        act.dry_radii = orig
    return s["t"], _blur(s["Tr"], 0.02, OUT_DT), float(np.max(s["S_max"])), float(np.max(s["n_act"]))


def eqnum(r_g, s_g, nb):
    return r_g * s_g ** norm.ppf((np.arange(nb) + 0.5) / nb)


def logbins(r_g, s_g, nb=60, m=600):
    z = np.linspace(-4, 4, nb)
    w = norm.pdf(z); w /= w.sum()
    k = np.round(w * m).astype(int)
    return np.repeat(r_g * s_g ** z, k)


def window(t, tr):
    i = np.argmax(tr < 0.995)
    return (t >= t[i] - 0.1) & (t <= t[i] + 0.9)


CASES = [  # (label, qw [kg/kg], N [cm-3], r_g, sigma_g, kappa)
    ("nominal, N=500", 7e-3, 500, 40e-9, 1.8, 0.3),
    ("nominal, N=2000", 7e-3, 2000, 40e-9, 1.8, 0.3),
    ("nominal, N=8000", 7e-3, 8000, 40e-9, 1.8, 0.3),
    ("dry, N=8000", 5.8e-3, 8000, 40e-9, 1.8, 0.3),
    ("small/broad/low-kappa, N=8000", 5.8e-3, 8000, 20e-9, 2.2, 0.1),
    ("small/broad/low-kappa, N=2000", 7e-3, 2000, 20e-9, 2.2, 0.1),
    ("humid, N=8000", 9.2e-3, 8000, 40e-9, 1.8, 0.3),
]
print(f"{'case':32s} {'Smax%':>6s} {'act12':>6s} {'actRef':>6s} {'dTr12':>8s} {'dTrLog':>8s} {'dN12%':>7s} {'dNlog%':>7s}")
for lab, qw, n, rg, sg, kap in CASES:
    nom = dict(r_g=rg, sigma_g=sg, kappa=kap, alpha_c=0.3)
    t0 = time.time()
    tR, trR, smax, actR = run(qw, n, eqnum(rg, sg, 96), nom)
    t1, tr1, _, act1 = run(qw, n, eqnum(rg, sg, 12), nom)
    tL, trL, _, actL = run(qw, n, logbins(rg, sg), nom)
    m = window(tR, trR)
    d12 = np.max(np.abs(tr1[m] - trR[m])); dL = np.max(np.abs(trL[m] - trR[m]))
    def cost(lg):
        _, tr, _, _ = run(qw, 10 ** lg, eqnum(rg, sg, 12), nom)
        return np.sqrt(np.mean((tr[m] - trR[m]) ** 2))
    r = minimize_scalar(cost, bounds=(np.log10(n) - 0.05, np.log10(n) + 0.05), method="bounded",
                        options=dict(xatol=1e-4))
    def costL(lg):
        _, tr, _, _ = run(qw, 10 ** lg, eqnum(rg, sg, 12), nom)
        return np.sqrt(np.mean((tr[m] - trL[m]) ** 2))
    rL = minimize_scalar(costL, bounds=(np.log10(n) - 0.05, np.log10(n) + 0.05), method="bounded",
                         options=dict(xatol=1e-4))
    print(f"{lab:32s} {(smax-1)*100:6.1f} {act1:6.3f} {actR:6.3f} {d12:8.5f} {dL:8.5f} "
          f"{(10**r.x/n-1)*100:7.2f} {(10**rL.x/n-1)*100:7.2f}  ({time.time()-t0:.0f} s)", flush=True)
