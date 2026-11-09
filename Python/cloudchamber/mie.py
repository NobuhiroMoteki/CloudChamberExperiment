# -*- coding: utf-8 -*-
"""Extinction efficiency lookup table built with MieScat_Py (github.com/NobuhiroMoteki/MieScat_Py).

MieScat_Py is vendored in Python/external/MieScat_Py.
The forward model assumes monodisperse droplets, but real droplets have a small size spread,
which washes out the narrow Mie resonances. The table is therefore averaged over a Gaussian
radius distribution of relative width SIGMA_REL (the absorption of water at 670 nm is far too
weak to damp them).

forward=True gives the effective Qext seen by the photodiode, which also collects the light scattered
into the cone it subtends: Qext - Qsca(theta < theta_PD(D)), averaged along the cloud from
D = PD_DIST - CLOUD_L/2 to PD_DIST + CLOUD_L/2 (distance from the PD). The PD (Hamamatsu S1223-01,
3.6 x 3.6 mm) sits at the flask wall, 100 mm from the centre, with nothing in front of it but the wall; the square is replaced by
the circle of equal area. Single scattering (Tr > 0.85 within the inference window), refraction by the flask wall neglected.
"""
import math
import os
import sys

import numpy as np
from numba import njit

from .constants import K_WATER, N_WATER, WAVELENGTH

_MIESCAT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "external", "MieScat_Py")
if _MIESCAT_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MIESCAT_DIR))
from miescat import miescat  # noqa: E402
from _mie_core import _compute_bessel_and_dd  # noqa: E402

# radius grid of the lookup table: uniform in a, so the Mie ripples are resolved (dx ~ 0.01)
A_MAX = 30e-6  # [m]
A_STEP = 1e-9  # [m]
SIGMA_REL = 0.015  # relative standard deviation of droplet radius

PD_SIDE = 3.6e-3  # [m] side of the square photodiode (S1223-01 datasheet; 3 mm before 2026-11-03)
PD_DIST = 0.100  # [m] flask centre to photodiode
CLOUD_L = 0.18  # [m] cloud along the beam, centred in the flask (= sbi_shape3.L_PATH3)
N_CONE = 1000  # angle samples of the forward cone

_CACHE = os.path.join(os.path.dirname(__file__), "qext_table_raw.npy")
_CACHE_FWD = os.path.join(os.path.dirname(__file__), "qext_table_raw_fwd.npy")


def qext(a, wavelength=WAVELENGTH, m_real=N_WATER, m_imag=K_WATER):
    """Qext of a water droplet of radius a [m] in air."""
    return miescat(wavelength, 1.0, 2.0 * a, m_real, m_imag, nang=2)[1]


def _cone_angles(n_d=400):
    """Half angles [rad] of the PD seen from n_d points evenly along the cloud."""
    d = np.linspace(PD_DIST - CLOUD_L / 2, PD_DIST + CLOUD_L / 2, n_d)
    return np.arctan(PD_SIDE / np.sqrt(np.pi) / d)


def qext_forward(a, wavelength=WAVELENGTH, m_real=N_WATER, m_imag=K_WATER):
    """Effective Qext (see module docstring) of a droplet of radius a [m]."""
    cone = _cone_angles()
    th = np.linspace(0.0, cone.max(), N_CONE)
    x = 2.0 * np.pi * a / wavelength
    m = m_real + 1j * m_imag
    nstop = int(np.floor(x + 4.0 * x ** 0.3333 + 2))
    # Mie coefficients a_n, b_n as in MieScat_Py (BH83 Eq. 4.88)
    dd, psi, xi = _compute_bessel_and_dd(x, m, nstop)
    nn = np.arange(1, nstop + 1)
    an = ((dd[1:] / m + nn / x) * psi[1:] - psi[:-1]) / ((dd[1:] / m + nn / x) * xi[1:] - xi[:-1])
    bn = ((m * dd[1:] + nn / x) * psi[1:] - psi[:-1]) / ((m * dd[1:] + nn / x) * xi[1:] - xi[:-1])
    q_ext = 2.0 / x ** 2 * np.sum((2 * nn + 1) * np.real(an + bn))
    # amplitudes S1, S2 (BH83 Eq. 4.74) on the cone angles
    mu = np.cos(th)
    pi_prev, pi_n = np.zeros_like(mu), np.ones_like(mu)
    s1 = np.zeros_like(mu, dtype=complex)
    s2 = np.zeros_like(mu, dtype=complex)
    for n in range(1, nstop + 1):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        f = (2 * n + 1) / (n * (n + 1))
        s1 += f * (an[n - 1] * pi_n + bn[n - 1] * tau_n)
        s2 += f * (an[n - 1] * tau_n + bn[n - 1] * pi_n)
        pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
    dq = (np.abs(s1) ** 2 + np.abs(s2) ** 2) / x ** 2 * np.sin(th)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dq[1:] + dq[:-1]) * np.diff(th))])
    return float(q_ext - np.interp(cone, th, cum).mean())


def qext_table_raw(forward=False):
    """Monodisperse Qext (effective Qext if forward) on the radius grid a = i*A_STEP (i = 0..),
    cached to qext_table_raw.npy (qext_table_raw_fwd.npy)."""
    path = _CACHE_FWD if forward else _CACHE
    if os.path.exists(path):
        return np.load(path)
    a_grid = np.arange(0.0, A_MAX + A_STEP, A_STEP)
    table = np.zeros(a_grid.size)
    table[1:] = [qext_forward(a) if forward else qext(a) for a in a_grid[1:]]
    bad = ~np.isfinite(table)  # isolated radii where the Bessel recursion divides by zero
    table[bad] = np.interp(np.nonzero(bad)[0], np.nonzero(~bad)[0], table[~bad])
    np.save(path, table)
    return table


@njit(cache=True)
def _gauss_average(raw, sigma_rel):
    # extinction cross section a^2*Qext is averaged, then divided by a^2 again
    n = raw.size
    out = raw.copy()
    for i in range(1, n):
        s = sigma_rel * i  # in grid units
        if s < 1.0:
            continue
        half = int(4.0 * s)
        num = 0.0
        den = 0.0
        for j in range(max(1, i - half), min(n, i + half + 1)):
            w = np.exp(-0.5 * ((j - i) / s) ** 2)
            num += w * j * j * raw[j]
            den += w * j * j
        out[i] = num / den
    return out


def qext_table(sigma_rel=SIGMA_REL, forward=False):
    """Qext table averaged over a Gaussian radius spread (sigma_rel=0 returns the raw table)."""
    raw = qext_table_raw(forward)
    if sigma_rel <= 0:
        return raw
    return _gauss_average(raw, sigma_rel)


SIGMA_GRID = np.array([0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15])
_STACK_CACHE = os.path.join(os.path.dirname(__file__), "qext_table_stack.npy")
_STACK_CACHE_FWD = os.path.join(os.path.dirname(__file__), "qext_table_stack_fwd.npy")


def qext_table_stack(forward=False):
    """Tables for every SIGMA_GRID value (rows), cached to qext_table_stack.npy
    (qext_table_stack_fwd.npy); used when the droplet-size spread changes with time (narrowing by
    condensational growth)."""
    path = _STACK_CACHE_FWD if forward else _STACK_CACHE
    if os.path.exists(path):
        return np.load(path)
    stack = np.array([qext_table(s, forward) for s in SIGMA_GRID])
    np.save(path, stack)
    return stack


@njit(cache=True)
def qext_lookup2(a, sigma_rel, stack, log_grid):
    """Qext at radius a [m] and relative spread sigma_rel, linear in a and in log(sigma_rel)
    between the rows of stack (clipped to the grid); a one-row stack ignores sigma_rel."""
    n = log_grid.size
    if n == 1:
        return qext_lookup(a, stack[0])
    x = math.log(max(sigma_rel, 1e-12))
    if x <= log_grid[0]:
        return qext_lookup(a, stack[0])
    if x >= log_grid[n - 1]:
        return qext_lookup(a, stack[n - 1])
    k = 0
    while log_grid[k + 1] < x:
        k += 1
    w = (x - log_grid[k]) / (log_grid[k + 1] - log_grid[k])
    return (1.0 - w) * qext_lookup(a, stack[k]) + w * qext_lookup(a, stack[k + 1])


@njit(cache=True)
def qext_lookup(a, table):
    """Linear interpolation of the Qext table at radius a [m]."""
    u = a / A_STEP
    i = int(u)
    if i >= table.size - 1:
        return table[table.size - 1]
    w = u - i
    return (1.0 - w) * table[i] + w * table[i + 1]
