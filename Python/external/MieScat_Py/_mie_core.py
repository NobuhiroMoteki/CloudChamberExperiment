# -*- coding: utf-8 -*-
"""
Internal helper module for Mie scattering calculations.

Shared computational routines used by miescat.py, miescat_charged.py,
and mie_complex_amplitudes.py.
"""

import numpy as np


def _compute_bessel_and_dd(x, m_r, nstop):
    """Compute Ricatti-Bessel functions and logarithmic derivative.

    Parameters
    ----------
    x : float
        Size parameter (real).
    m_r : complex
        Relative refractive index of particle (m_p / m_m).
    nstop : int
        Number of expansion terms.

    Returns
    -------
    DD : ndarray, complex, shape (nstop+1,)
        Logarithmic derivative D_n(m_r * x).
    psi : ndarray, float, shape (nstop+1,)
        Ricatti-Bessel function of the first kind: x * j_n(x).
    xi : ndarray, complex, shape (nstop+1,)
        Ricatti-Bessel function of the third kind: x * (j_n(x) + i*y_n(x)).
    """
    y = m_r * x
    nmx = int(np.floor(max(nstop, abs(y)) + 15))

    # Logarithmic derivative D_n(y) by downward recurrence (BH83)
    DD = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        DD[n-1] = n/y - 1.0/(DD[n] + n/y)
    DD = DD[:nstop + 1]

    # psi_n(x) = x * j_n(x), computed via ratio R_n = psi_n / psi_{n-1}
    # using downward recurrence (Mishchenko et al. 2002)
    R = np.zeros(nmx + 1)
    R[nmx] = x / (2*nmx + 1)
    for n in range(nmx - 1, -1, -1):
        R[n] = 1.0 / ((2*n + 1)/x - R[n+1])
    psi = np.zeros(nstop + 1)
    psi[0] = R[0] * np.cos(x)
    for n in range(1, nstop + 1):
        psi[n] = R[n] * psi[n-1]

    # chi_n(x) = -x * y_n(x), by forward recurrence
    chi = np.zeros(nstop + 1)
    chi[0] = -np.cos(x)
    chi[1] = chi[0]/x - np.sin(x)
    for n in range(2, nstop + 1):
        chi[n] = ((2*n - 1)/x)*chi[n-1] - chi[n-2]

    xi = psi + 1j*chi  # Ricatti-Bessel of the third kind
    return DD, psi, xi


def _compute_efficiencies_and_amplitudes(a, b, x, k, nstop, nang):
    """Compute optical efficiency factors and scattering amplitudes.

    Parameters
    ----------
    a, b : ndarray, complex, shape (nstop+1,)
        Partial wave coefficients (index 0 unused).
    x : float
        Size parameter.
    k : float
        Wavenumber in the medium.
    nstop : int
        Number of expansion terms.
    nang : int
        Number of scattering angle grid points (0 to 180 deg).

    Returns
    -------
    Qsca, Qext, Qabs, Qback : float
        Optical efficiency factors.
    S11, S22 : ndarray, complex, shape (nang,)
        Scattering amplitude matrix elements (Mishchenko convention).
    """
    n_arr = np.arange(1, nstop + 1)
    fn1 = np.empty(nstop + 1)
    fn2 = np.empty(nstop + 1)
    sg  = np.empty(nstop + 1)
    fn1[0] = 0.0
    fn2[0] = 0.0
    sg[0]  = 0.0
    fn1[1:] = 2*n_arr + 1
    fn2[1:] = (2*n_arr + 1) / (n_arr * (n_arr + 1))
    sg[1:]  = (-1.0)**n_arr

    Qsca  = float((2.0/x**2) * np.sum(fn1 * (np.abs(a)**2 + np.abs(b)**2)))
    Qext  = float((2.0/x**2) * np.sum(fn1 * np.real(a + b)))
    Qabs  = Qext - Qsca
    Qback = float(np.abs(np.sum(fn1 * sg * (a - b)))**2 / x**2)

    # Angular functions pi_n(cos theta) and tau_n(cos theta)
    mu = np.cos(np.linspace(0.0, np.pi, nang))
    pi_n  = np.zeros((nstop + 1, nang))
    tau_n = np.zeros((nstop + 1, nang))
    pi_n[1, :]  = 1.0
    if nstop >= 2:
        pi_n[2, :]  = 3.0*mu
        tau_n[1, :] = mu
        tau_n[2, :] = 6.0*mu**2 - 3.0
        for n in range(3, nstop + 1):
            pi_n[n, :]  = ((2*n-1)/(n-1))*mu*pi_n[n-1, :] - (n/(n-1))*pi_n[n-2, :]
            tau_n[n, :] = n*mu*pi_n[n, :] - (n+1)*pi_n[n-1, :]
    else:
        tau_n[1, :] = mu

    # Scattering amplitudes S1, S2 — vectorized over angle axis
    fn2_col = fn2[:, np.newaxis]   # (nstop+1, 1)
    a_col   = a[:, np.newaxis]
    b_col   = b[:, np.newaxis]
    S1 = np.sum(fn2_col * (a_col*pi_n  + b_col*tau_n), axis=0)  # BH83, Eq.4.74
    S2 = np.sum(fn2_col * (a_col*tau_n + b_col*pi_n),  axis=0)  # BH83, Eq.4.74

    # Convert to Mishchenko amplitude matrix convention
    S11 = S2 / (-1j*k)
    S22 = S1 / (-1j*k)

    return Qsca, Qext, Qabs, Qback, S11, S22
