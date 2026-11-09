# -*- coding: utf-8 -*-
"""Forward model with aerosol activation (kappa-Koehler) and a dry-size distribution.

Differences from forward.py (the Docs model):
- Droplet equilibrium saturation ratio S_eq(a) = (a^3-ad^3)/(a^3-ad^3(1-kappa)) exp(A/a)
  (Petters and Kreidenweis 2007), so particles activate only above their critical supersaturation.
- Lognormal dry radius distribution (geometric mean r_g, geometric sd sigma_g) split into
  n_bins equal-number bins; the activated number follows from the supersaturation reached.
- Particles start as haze in equilibrium with the initial relative humidity.
- Growth law a da/dt = (S - S_eq)/(F_k + F_d) (Rogers and Yau 1989, Eq. 7.17) with gas-kinetic
  corrections of D_v and k_a (condensation coefficient alpha_c, thermal accommodation alpha_t).
- Each bin is integrated with explicit sub-steps sized by its relaxation rate (haze is stiff).
- D_v(T, p), k_a(T) and the surface tension sigma_w(T) follow the air state (Pruppacher and Klett).
- Vapour pressure e = q_v p/(eps + q_v) for the mixing ratio q_v, and the latent heating is
  -l_v dq_v/dt with the total pressure p (exact rewriting of the entropy equation; the
  former (R_v T - l_v) term came from approximating dp_d/p_d by dp/p).
Temperature, pressure profile, wall heating and optics are as in forward.py, except that the
droplet-size spread used to smooth Qext may narrow with growth and the optical path may be shorter
than the flask diameter (cloud-free layer near the wall).
"""
import math

import numpy as np
from numba import njit
from scipy.stats import norm

from .constants import CL, CPD, CPV, EPS, L_PATH, LV, RD, RHO_L, RV
from .forward import _es
from .mie import SIGMA_GRID, SIGMA_REL, qext_lookup2, qext_table, qext_table_stack

SIGMA_W = None  # [N m-1] surface tension of water; None: sigma_w(T)
A_SPREAD_REF = 0.69e-6  # [m] radius at which sigma_rel is given (first Qext maximum)

MODEL_VERSION = "activation-v2"


@njit(cache=True)
def _seq(a, ad3, kappa, A):
    a3 = a * a * a
    return (a3 - ad3) / (a3 - ad3 * (1.0 - kappa)) * math.exp(A / a)


@njit(cache=True)
def _dv_air(T, P):
    """Diffusivity of water vapour in air [m2 s-1] (Pruppacher and Klett 1997, Eq. 13.3)."""
    return 2.11e-5 * (T / 273.15) ** 1.94 * (101325.0 / P)


@njit(cache=True)
def _ka_air(T):
    """Thermal conductivity of air [W m-1 K-1] (Pruppacher and Klett 1997, Eq. 13.18a)."""
    return 4.1868e-3 * (5.69 + 0.017 * (T - 273.15))


@njit(cache=True)
def _sigma_w(T):
    """Surface tension of water against air [N m-1] (linear fit, about -40 to +40 C)."""
    return 0.0761 - 1.55e-4 * (T - 273.15)


@njit(cache=True)
def _dadt(a, S, ad3, kappa, A, T, P, rho_a, es, alpha_c, alpha_t):
    dv0 = _dv_air(T, P)
    ka0 = _ka_air(T)
    dv = dv0 / (1.0 + dv0 / (alpha_c * a) * math.sqrt(2.0 * math.pi / (RV * T)))
    ka = ka0 / (1.0 + ka0 / (alpha_t * a * rho_a * CPD) * math.sqrt(2.0 * math.pi / (RD * T)))
    fk = (LV / (RV * T) - 1.0) * LV * RHO_L / (ka * T)
    fd = RHO_L * RV * T / (dv * es)
    return (S - _seq(a, ad3, kappa, A)) / ((fk + fd) * a)


@njit(cache=True)
def _crit_radius(ad, kappa, A):
    """Radius at the maximum of S_eq(a) (critical radius), by golden-section search in ln a.

    Unlike the approximation sqrt(3 kappa ad^3 / A), valid for any kappa and dry size.
    """
    ad3 = ad ** 3
    lo, hi = math.log(ad * (1.0 + 1e-6)), math.log(ad * 1e4)
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(100):
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        if _seq(math.exp(x1), ad3, kappa, A) > _seq(math.exp(x2), ad3, kappa, A):
            hi = x2
        else:
            lo = x1
    return math.exp(0.5 * (lo + hi))


@njit(cache=True)
def _eq_radius(ad, kappa, A, S, a_crit):
    """Stable equilibrium radius S_eq(a) = S on (ad, a_crit), for S below the critical value."""
    ad3 = ad ** 3
    lo, hi = ad * (1.0 + 1e-12), a_crit
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _seq(mid, ad3, kappa, A) < S:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _haze_radius(ad, kappa, A, S0):
    """Equilibrium wet radius at S0 < S_crit (S_eq increases monotonically below a_crit)."""
    ad3 = ad ** 3
    lo = ad * (1.0 + 1e-9)
    hi = _crit_radius(ad, kappa, A)
    if _seq(hi, ad3, kappa, A) <= S0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _seq(mid, ad3, kappa, A) < S0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _simulate(qw, n_tot, ad, kappa, alpha_c, alpha_t, sigma_w, T_ini, p_ini, p_offset, tau_vac,
              frac1, tau_vac2, t_open, inv_tau_wall, t_wall_on, wall_sqrt, t_end, dt, out_every,
              stack, log_grid, sigma_rel, spread_exp, spread_floor, l_path, l_inner, l_wall_k):
    nb = ad.size
    w = n_tot / nb  # particles per kg in each equal-number bin
    ad3 = ad ** 3
    n_steps = int(round(t_end / dt)) + 1
    n_out = (n_steps - 1) // out_every + 1
    t_out = np.empty(n_out)
    p_out = np.empty(n_out)
    T_out = np.empty(n_out)
    S_out = np.empty(n_out)
    tr_out = np.empty(n_out)
    amax_out = np.empty(n_out)
    nact_out = np.empty(n_out)

    T = T_ini
    sw = sigma_w if sigma_w > 0.0 else _sigma_w(T)  # sigma_w <= 0: temperature-dependent
    A_ini = max(2.0 * sw / (RHO_L * RV * T), 1e-30)
    A = A_ini
    S0 = min(qw * p_ini / ((EPS + qw) * _es(T_ini)), 0.995)
    a = np.empty(nb)
    a_crit0 = np.empty(nb)  # critical radii at T_ini; scaled as A^-1/2 when A changes
    for j in range(nb):
        a[j] = _haze_radius(ad[j], kappa, A, S0)
        a_crit0[j] = _crit_radius(ad[j], kappa, A)
    a_crit = a_crit0.copy()

    s_max = 0.0
    for i in range(n_steps):
        t = i * dt
        # pumping conductance ramps up linearly while the ball valve opens (t_open);
        # afterwards p relaxes towards p_offset with a fraction frac1 decaying with tau_vac and the
        # rest with tau_vac2 (frac1 = 1: single exponential, Eq. 111)
        if t < t_open:
            G, g = 0.5 * t * t / t_open, t / t_open
        else:
            G, g = t - 0.5 * t_open, 1.0
        ex1 = math.exp(-G / tau_vac)
        ex2 = math.exp(-G / tau_vac2)
        P = (p_ini - p_offset) * (frac1 * ex1 + (1.0 - frac1) * ex2) + p_offset
        dP_dt = -(p_ini - p_offset) * g * (frac1 * ex1 / tau_vac + (1.0 - frac1) * ex2 / tau_vac2)
        rho_a = P / (RD * T)
        es = _es(T)
        sw = sigma_w if sigma_w > 0.0 else _sigma_w(T)
        A = max(2.0 * sw / (RHO_L * RV * T), 1e-30)
        a_crit = a_crit0 * math.sqrt(A_ini / A)  # a_crit ~ (3 kappa ad^3 / A)^1/2

        sum_a3 = 0.0
        ext = 0.0
        n_act = 0.0
        for j in range(nb):
            sum_a3 += a[j] ** 3
            # relative radius spread within the bin; diffusional growth keeps a*da constant, so it
            # narrows as (A_SPREAD_REF/a)^spread_exp down to spread_floor (other broadening)
            sr = max(sigma_rel * (A_SPREAD_REF / a[j]) ** spread_exp, spread_floor)
            ext += a[j] * a[j] * qext_lookup2(a[j], sr, stack, log_grid)
            if a[j] > a_crit[j]:
                n_act += 1.0
        ql = 4.0 * math.pi / 3.0 * RHO_L * w * sum_a3
        qv = qw - ql
        S = qv * P / (EPS + qv) / es
        s_max = max(s_max, S)
        l_eff = l_path
        if l_wall_k > 0.0:  # cloud-free layer near the wall thickening as sqrt(t) (wall heating)
            l_eff = max(min(l_path, l_inner - 2.0 * l_wall_k * math.sqrt(t)), 0.0)
        Tr = math.exp(-math.pi * ext * w * rho_a * l_eff)

        if i % out_every == 0:
            k = i // out_every
            t_out[k] = t
            p_out[k] = P
            T_out[k] = T
            S_out[k] = S
            tr_out[k] = Tr
            amax_out[k] = a[nb - 1]
            nact_out[k] = n_act / nb

        d_sum_a3 = 0.0
        for j in range(nb):
            x = a[j]
            amin = ad[j] * (1.0 + 1e-9)
            dist = max(x - ad[j], 1e-3 * ad[j])
            f = _dadt(x, S, ad3[j], kappa, A, T, P, rho_a, es, alpha_c, alpha_t)
            h = 0.01 * dist  # S_eq varies on the scale of (a - ad), tiny for small/weakly soluble
            lam = abs(_dadt(x + h, S, ad3[j], kappa, A, T, P, rho_a, es, alpha_c, alpha_t) - f) / h
            if x < a_crit[j] and S < _seq(a_crit[j], ad3[j], kappa, A) and lam * dt > 1.0:
                # unactivated haze relaxing much faster than dt: quasi-equilibrium with S
                x = _eq_radius(ad[j], kappa, A, S, a_crit[j])
            else:
                nsub = min(int(dt * lam / 0.5) + 1, 2000)
                ds = dt / nsub
                for _ in range(nsub):
                    x += ds * _dadt(x, S, ad3[j], kappa, A, T, P, rho_a, es, alpha_c, alpha_t)
                    if x < amin:
                        x = amin
            d_sum_a3 += x ** 3 - a[j] ** 3
            a[j] = x

        dqv_dt = -4.0 * math.pi / 3.0 * RHO_L * w * d_sum_a3 / dt
        cp = CPD + qv * CPV + ql * CL
        dT_dt = (-LV * dqv_dt + T * (RV * qv + RD) * dP_dt / P) / cp
        if t >= t_wall_on:
            dT_dt += (T_ini - T) * inv_tau_wall
        # conduction through a thermal boundary layer growing as sqrt(t): rate h/sqrt(t)
        dT_dt += (T_ini - T) * wall_sqrt / math.sqrt(max(t, dt))
        T += dt * dT_dt
    return t_out, p_out, T_out, S_out, tr_out, amax_out, nact_out, s_max


_TABLES = {}


def dry_radii(r_g, sigma_g, n_bins):
    """Dry radii of n_bins equal-number bins of a lognormal distribution."""
    z = norm.ppf((np.arange(n_bins) + 0.5) / n_bins)
    return r_g * sigma_g ** z


def dry_radii_bimodal(r_g, sigma_g, frac2, r_g2, sigma_g2, n_bins):
    """Equal-number bins of a two-mode lognormal mixture (number fraction frac2 in mode 2)."""
    q = (np.arange(n_bins) + 0.5) / n_bins
    lo = np.log(min(r_g, r_g2)) - 6 * np.log(max(sigma_g, sigma_g2))
    hi = np.log(max(r_g, r_g2)) + 6 * np.log(max(sigma_g, sigma_g2))
    x = np.linspace(lo, hi, 4001)
    cdf = ((1 - frac2) * norm.cdf((x - np.log(r_g)) / np.log(sigma_g))
           + frac2 * norm.cdf((x - np.log(r_g2)) / np.log(sigma_g2)))
    return np.exp(np.interp(q, cdf, x))


def simulate_activation(qw, n_tot, r_g=40e-9, sigma_g=1.8, kappa=0.3, alpha_c=1.0, alpha_t=1.0,
                        n_bins=16, T_ini=299.86, p_ini=1002e2, p_offset=441.34e2, tau_vac=3.9474,
                        frac1=1.0, tau_vac2=10.0, t_open=0.0, tau_wall=np.inf, t_wall_on=0.0,
                        wall_sqrt=0.0, t_end=4.0, dt=1e-3, out_dt=1e-2, sigma_rel=SIGMA_REL,
                        sigma_w=SIGMA_W, spread_exp=0.0, spread_floor=0.0, l_path=L_PATH, mode2=None,
                        forward=False, l_inner=0.0, l_wall_k=0.0):
    """Simulate one decompression with activation microphysics.

    qw [kg/kg] total water, n_tot [#/kg] total aerosol number (all sizes, what a CPC counts),
    r_g [m] and sigma_g: lognormal dry-radius distribution, kappa: hygroscopicity,
    alpha_c / alpha_t: condensation / thermal accommodation coefficients.
    Pressure: p_offset + (p_ini - p_offset)*(frac1*exp(-G/tau_vac) + (1-frac1)*exp(-G/tau_vac2)),
    G = integrated pumping conductance; t_open [s]: valve opening time (conductance ramps linearly
    from 0 to 1). frac1 = 1 gives the single exponential of Eq. (111).
    Wall heating: Newtonian (T_ini - T)/tau_wall from t_wall_on, and/or boundary-layer
    conduction (T_ini - T)*wall_sqrt/sqrt(t) [wall_sqrt in s^-1/2].
    Optics: Qext is averaged over a relative radius spread sigma_rel within each bin; with
    spread_exp > 0 the spread is sigma_rel at a = A_SPREAD_REF and narrows as (A_SPREAD_REF/a)^spread_exp
    (2 for diffusional growth) down to spread_floor. l_path [m]: effective optical path in the cloud.
    l_wall_k > 0 [m s^-1/2] (sensitivity tests only): the cloud-free layer near the wall grows as
    l_wall_k*sqrt(t) on each side, so that the path is min(l_path, l_inner - 2*l_wall_k*sqrt(t)).
    mode2 = (frac2, r_g2, sigma_g2): optional second lognormal mode holding the number fraction
    frac2 of n_tot (e.g. a nucleation mode below the CPC cut-off).
    forward: use the effective Qext that includes the forward-scattered light reaching the photodiode
    (mie.qext_forward).
    Returns arrays sampled every out_dt plus 'n_act' (activated fraction) and 'S_max'.
    """
    if spread_exp == 0.0:
        if (sigma_rel, forward) not in _TABLES:
            _TABLES[(sigma_rel, forward)] = qext_table(sigma_rel, forward)[None, :]
        stack, log_grid = _TABLES[(sigma_rel, forward)], np.zeros(1)
    else:
        if ("stack", forward) not in _TABLES:
            _TABLES[("stack", forward)] = qext_table_stack(forward)
        stack, log_grid = _TABLES[("stack", forward)], np.log(SIGMA_GRID)
    out_every = max(1, int(round(out_dt / dt)))
    ad = dry_radii(r_g, sigma_g, n_bins) if mode2 is None else dry_radii_bimodal(r_g, sigma_g, *mode2, n_bins)
    t, p, T, S, tr, amax, nact, s_max = _simulate(
        qw, n_tot, ad, kappa, alpha_c, alpha_t, -1.0 if sigma_w is None else sigma_w, T_ini, p_ini, p_offset, tau_vac,
        frac1, tau_vac2, t_open, 1.0 / tau_wall, t_wall_on, wall_sqrt, t_end, dt, out_every,
        stack, log_grid, sigma_rel, spread_exp, spread_floor, l_path, l_inner, l_wall_k)
    return {"t": t, "p": p, "T": T, "S": S, "Tr": tr, "a_max": amax, "n_act": nact,
            "S_max": s_max}
