# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:00:56 2022

@author: Moteki
"""

import numpy as np
from _mie_core import _compute_bessel_and_dd, _compute_efficiencies_and_amplitudes


def mie_complex_amplitudes(wl_0, m_m, d_p, m_p_real, m_p_imag, nang=3):
    """Complex scattering amplitudes of a single homogeneous sphere.

    Calculates the forward and backward complex scattering amplitudes
    observable by the Complex Amplitude Sensing (CAS) protocol, based on
    Bohren and Huffman (1983) [BH83] and Fu and Sun (2001).

    Theoretical assumptions
    -----------------------
    1. Gaussian units for mathematical expressions.
    2. Surrounding medium must be nonabsorbing and nonmagnetic.
    3. Particle must be nonmagnetic.

    Computational assumptions
    -------------------------
    Number of VSWF expansion terms nstop = floor(x + 4*x^(1/3) + 2) (BH83).

    Parameters
    ----------
    wl_0 : float
        Wavelength in vacuum (= c / omega).
    m_m : float
        Real refractive index of the surrounding medium.
    d_p : float
        Particle diameter.
    m_p_real : float
        Real part of the particle refractive index.
    m_p_imag : float
        Imaginary part of the particle refractive index.
    nang : int, optional
        Number of scattering angle grid points (0 to 180 deg). Default 3.

    Returns
    -------
    S_fwd_real : float
        Real part of the forward scattering amplitude S(0) [length].
    S_fwd_imag : float
        Imaginary part of the forward scattering amplitude S(0) [length].
    S_bak_real : float
        Real part of the backward scattering amplitude S_bak [length].
    S_bak_imag : float
        Imaginary part of the backward scattering amplitude S_bak [length].

    References
    ----------
    Moteki 2021, Optics Express, https://doi.org/10.1364/OE.423175
    Moteki and Adachi 2024, Optics Express, https://doi.org/10.1364/OE.533776
    """
    m_p = m_p_real + 1j*m_p_imag
    k0  = 2.0*np.pi / wl_0
    k   = m_m * k0
    x   = k * d_p / 2.0
    m_r = m_p / m_m

    nstop = int(np.floor(abs(x) + 4.0*abs(x)**0.3333 + 2))

    DD, psi, xi = _compute_bessel_and_dd(x, m_r, nstop)

    # Partial wave coefficients a_n, b_n for a neutral sphere (BH83, Eq.4.88)
    n_arr    = np.arange(1, nstop + 1, dtype=float)
    n_over_x = n_arr / x
    a = np.zeros(nstop + 1, dtype=complex)
    b = np.zeros(nstop + 1, dtype=complex)
    a[1:] = ((DD[1:]/m_r + n_over_x)*psi[1:] - psi[:-1]) \
          / ((DD[1:]/m_r + n_over_x)*xi[1:]  - xi[:-1])
    b[1:] = ((m_r*DD[1:] + n_over_x)*psi[1:] - psi[:-1]) \
          / ((m_r*DD[1:] + n_over_x)*xi[1:]  - xi[:-1])

    _, _, _, _, S11, S22 = \
        _compute_efficiencies_and_amplitudes(a, b, x, k, nstop, int(nang))

    S_fwd = (S11[0] + S22[0]) / 2.0
    S_bak = (-S11[-1] + S22[-1]) / np.sqrt(2.0)

    return S_fwd.real, S_fwd.imag, S_bak.real, S_bak.imag
