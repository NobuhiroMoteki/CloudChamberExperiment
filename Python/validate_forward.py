# -*- coding: utf-8 -*-
"""Check cloudchamber.forward against the original pure-Python implementation (Python/Master)."""
import importlib.util
import math
import os
import time

import numpy as np

# original course code, loaded under another name (MieScat_Py also provides a module "miescat")
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("miescat_master", os.path.join(HERE, "Master", "miescat.py"))
miescat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(miescat)

from cloudchamber import forward  # noqa: E402
from cloudchamber.constants import (A_ES, B_ES, CL, CPD, CPV, DV, EPS, KA,  # noqa: E402
                                    K_WATER, L_PATH, LV, N_WATER, RD, RHO_L, RV)
from cloudchamber.features import zeta_features  # noqa: E402
from cloudchamber.mie import qext_lookup, qext_table  # noqa: E402

EPER_W = (N_WATER + 1j * K_WATER) ** 2


def qext_orig(a, mper_p):
    return miescat.gMIE_NonAbsMed(wl_0=670e-9, r_p=a, eper_m=1 + 0j, mper_m=1 + 0j,
                                  eper_p=EPER_W, mper_p=mper_p)[1]


def reference(qw, nd, mper_p, T_ini=299.86, p_ini=1002e2, p_offset=441.34e2, tau_vac=3.9474):
    """Original CloudChamberForwardCalc loop, Tr sampled every 10 ms."""
    es = lambda T: A_ES * math.exp(-B_ES / T)
    T, a, dt = T_ini, 50e-9, 1e-3
    ts, ps, trs = [], [], []
    for i in range(4001):
        t = i * dt
        P = (p_ini - p_offset) * math.exp(-t / tau_vac) + p_offset
        dP_dt = -(p_ini - p_offset) / tau_vac * math.exp(-t / tau_vac)
        n = nd * P / (RD * T)
        Tr = math.exp(-math.pi * a ** 2 * qext_orig(a, mper_p) * n * L_PATH)
        if i % 10 == 0:
            ts.append(t); ps.append(P); trs.append(Tr)
        ql = 4 * math.pi / 3 * RHO_L * a ** 3 * nd
        qv = qw - ql
        S = qv * P / EPS / es(T)
        if S >= 1:
            AA, BB, CC = LV * DV / RV, S * es(T) / T, KA * T
            x = T
            while True:
                f = -AA * es(x) / x - KA * x + AA * BB + CC
                df = -AA * A_ES / x ** 2 * (B_ES / x - 1) * math.exp(-B_ES / x) - KA
                x1 = x - f / df
                if abs(x1 - x) < 1e-7:  # original: 1e-3 and returns x, not x1
                    x = x1
                    break
                x = x1
            da_dt = (S * es(T) / T - es(x) / x) * DV / (RHO_L * RV) / a
        else:
            da_dt = 0
        dqv_dt = -4 * math.pi * RHO_L * nd * a ** 2 * da_dt
        cp = CPD + qv * CPV + ql * CL
        T += dt * ((RV * T - LV) * dqv_dt + T * (RV * qv + RD) * dP_dt / P) / cp
        a += dt * da_dt
    return np.array(ts), np.array(ps), np.array(trs)


def main():
    table = qext_table(0.0)

    print("1) raw Qext table (MieScat_Py) vs Master/miescat (m=1.33+2e-8i, mu=1)")
    radii = [0.0503e-6, 0.3005e-6, 0.7005e-6, 1.2345e-6, 2.5005e-6, 4.3215e-6, 9.8705e-6]  # off-grid
    for a in radii:
        q_ref = qext_orig(a, 1 + 0j)
        q_new = qext_lookup(a, table)
        q_bug = qext_orig(a, EPER_W)
        print(f"   a={a*1e6:6.3f} um  ref={q_ref:.5f}  table={q_new:.5f}  "
              f"rel.err={abs(q_new-q_ref)/q_ref:.1e}   (old buggy mu: {q_bug:.4f})")

    print("2) full simulation vs original loop (corrected mu); 'raw' = monodisperse table, "
          "'smooth' = default table averaged over the radius spread")
    cases = [(0.010, 1e9), (0.013, 1e10), (0.016, 1e11)]
    for qw, nd in cases:
        t0 = time.perf_counter()
        t, p, tr_ref = reference(qw, nd, 1 + 0j)
        t_ref = time.perf_counter() - t0
        raw = forward.simulate(qw, nd, sigma_rel=0.0)
        smooth = forward.simulate(qw, nd)
        z_ref = zeta_features(t, p, tr_ref)[2:]
        z_raw = zeta_features(raw["t"], raw["p"], raw["Tr"])[2:]
        z_smo = zeta_features(smooth["t"], smooth["p"], smooth["Tr"])[2:]
        _, _, tr_bug = reference(qw, nd, EPER_W)
        z_bug = zeta_features(t, p, tr_bug)[2:]
        print(f"   qw={qw*1e3:.0f} g/kg nd=1e{math.log10(nd):.0f}: "
              f"max|dTr| raw={np.max(np.abs(raw['Tr']-tr_ref)):.1e} "
              f"smooth={np.max(np.abs(smooth['Tr']-tr_ref)):.1e}  (ref run {t_ref:.1f} s)\n"
              f"      zeta ref={z_ref[0]:.4f},{z_ref[1]:.4f} raw={z_raw[0]:.4f},{z_raw[1]:.4f} "
              f"smooth={z_smo[0]:.4f},{z_smo[1]:.4f} old-buggy={z_bug[0]:.4f},{z_bug[1]:.4f}")

    print("3) speed")
    forward.simulate(0.013, 1e10)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        forward.simulate(0.013, 1e10)
    print(f"   {(time.perf_counter()-t0)/n*1e3:.2f} ms per simulation")


if __name__ == "__main__":
    main()
