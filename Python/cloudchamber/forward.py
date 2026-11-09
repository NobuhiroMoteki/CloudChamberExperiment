# -*- coding: utf-8 -*-
"""Forward model of the decompressed cloud chamber (KankyoJissu2022 Sec. 5.1).

Same equations and explicit Euler scheme as CloudChamberSimulator.py, compiled with numba.
"""
import math

import numpy as np
from numba import njit

from .constants import (A_ES, B_ES, CL, CPD, CPV, DV, EPS, KA, L_PATH, LV,
                        RD, RHO_L, RV)
from .mie import SIGMA_REL, qext_lookup, qext_table


@njit(cache=True)
def _es(T):
    return A_ES * math.exp(-B_ES / T)


@njit(cache=True)
def _simulate(qw, nd, T_ini, p_ini, p_offset, tau_vac, a_ini, inv_tau_wall, t_wall_on,
              t_end, dt, out_every, table):
    n_steps = int(round(t_end / dt)) + 1
    n_out = (n_steps - 1) // out_every + 1
    t_out = np.empty(n_out)
    p_out = np.empty(n_out)
    T_out = np.empty(n_out)
    S_out = np.empty(n_out)
    a_out = np.empty(n_out)
    tr_out = np.empty(n_out)

    T = T_ini
    a = a_ini
    AA = LV * DV / RV
    for i in range(n_steps):
        t = i * dt
        # pressure profile, Eq. (111)
        ex = math.exp(-t / tau_vac)
        P = (p_ini - p_offset) * ex + p_offset
        dP_dt = -(p_ini - p_offset) / tau_vac * ex

        # transmittance, Eq. (115)
        n = nd * P / (RD * T)
        Tr = math.exp(-math.pi * a * a * qext_lookup(a, table) * n * L_PATH)

        ql = 4.0 * math.pi / 3.0 * RHO_L * a ** 3 * nd
        qv = qw - ql
        e = qv * P / (EPS + qv)  # qv: mixing ratio
        S = e / _es(T)

        if i % out_every == 0:
            k = i // out_every
            t_out[k] = t
            p_out[k] = P
            T_out[k] = T
            S_out[k] = S
            a_out[k] = a
            tr_out[k] = Tr

        # droplet growth, Eqs. (102)-(103); droplet temperature T0 by Newton method
        if S >= 1.0:
            BB = S * _es(T) / T
            CC = KA * T
            x = T
            for _ in range(50):
                f = -AA * _es(x) / x - KA * x + AA * BB + CC
                df = -AA * A_ES / x ** 2 * (B_ES / x - 1.0) * math.exp(-B_ES / x) - KA
                x1 = x - f / df
                if abs(x1 - x) < 1e-7:
                    x = x1
                    break
                x = x1
            T0 = x
            da_dt = (S * _es(T) / T - _es(T0) / T0) * DV / (RHO_L * RV) / a
        else:
            da_dt = 0.0

        # temperature tendency, Eqs. (71), (74), (114)
        dqv_dt = -4.0 * math.pi * RHO_L * nd * a * a * da_dt
        cp = CPD + qv * CPV + ql * CL
        dT_dt = (-LV * dqv_dt + T * (RV * qv + RD) * dP_dt / P) / cp
        # heat from the chamber wall (kept at T_ini), effective relaxation time tau_wall,
        # switched on t_wall_on after the start of decompression
        if t >= t_wall_on:
            dT_dt += (T_ini - T) * inv_tau_wall

        T += dt * dT_dt
        a += dt * da_dt
    return t_out, p_out, T_out, S_out, a_out, tr_out


_TABLES = {}


def simulate(qw, nd, T_ini=299.86, p_ini=1002e2, p_offset=441.34e2, tau_vac=3.9474,
             a_ini=50e-9, tau_wall=np.inf, t_wall_on=0.0, t_end=4.0, dt=1e-3, out_dt=1e-2,
             sigma_rel=SIGMA_REL):
    """Simulate one decompression.

    qw [kg/kg], nd [#/kg], T_ini [K], p_ini and p_offset [Pa], tau_vac [s], a_ini [m].
    tau_wall [s]: effective time constant of heating by the wall (inf = adiabatic, Docs model),
    active from t_wall_on [s] after the start of decompression.
    sigma_rel: relative radius spread used to smooth Qext (0 = monodisperse).
    Returns a dict of arrays sampled every out_dt (10 ms = DAQ sampling interval).
    """
    if sigma_rel not in _TABLES:
        _TABLES[sigma_rel] = qext_table(sigma_rel)
    out_every = max(1, int(round(out_dt / dt)))
    t, p, T, S, a, tr = _simulate(qw, nd, T_ini, p_ini, p_offset, tau_vac, a_ini,
                                  1.0 / tau_wall, t_wall_on, t_end, dt, out_every,
                                  _TABLES[sigma_rel])
    return {"t": t, "p": p, "T": T, "S": S, "a": a, "Tr": tr}
