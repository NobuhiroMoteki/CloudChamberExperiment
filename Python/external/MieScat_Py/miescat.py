# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:00:56 2022

@author: Moteki
"""

import numpy as np
from _mie_core import _compute_bessel_and_dd, _compute_efficiencies_and_amplitudes


def miescat(wl_0, m_m, d_p, m_p_real, m_p_imag, dens_p=1.0, nang=3):
    """Mie scattering properties of a single homogeneous sphere.

    Calculates light scattering properties based on the mathematical
    formulations of Bohren and Huffman (1983) [BH83] and Fu and Sun (2001).

    Theoretical assumptions
    -----------------------
    1. Gaussian units for mathematical expressions.
    2. Surrounding medium must be nonabsorbing (required to define Qsca and
       Qabs without ambiguity); it may be magnetic.
    3. Particle may be absorbing and magnetic.

    Computational assumptions
    -------------------------
    Number of VSWF expansion terms nstop = floor(x + 4*x^(1/3) + 2) (BH83).

    Parameters
    ----------
    wl_0 : float
        Wavelength in vacuum (= c / omega) [m].
    m_m : float
        Real refractive index of the surrounding medium.
    d_p : float
        Particle diameter [m].
    m_p_real : float
        Real part of the particle refractive index.
    m_p_imag : float
        Imaginary part of the particle refractive index.
    dens_p : float, optional
        Particle density [g/cm^3]. Default 1.0.
    nang : int, optional
        Number of scattering angle grid points (0 to 180 deg). Default 3.

    Returns
    -------
    Qsca : float
        Scattering efficiency Csca / (pi * r_p^2).
    Qext : float
        Extinction efficiency Cext / (pi * r_p^2).
    Qabs : float
        Absorption efficiency Cabs / (pi * r_p^2).
    Qback : float
        Backscattering efficiency.
    S11 : ndarray, complex, shape (nang,)
        (1,1) element of the 2x2 amplitude scattering matrix (BH83 Eq.3.12).
    S22 : ndarray, complex, shape (nang,)
        (2,2) element of the 2x2 amplitude scattering matrix (BH83 Eq.3.12).
    MSC : float
        Mass scattering cross section Csca / mass_p [m^2/g].
    MEC : float
        Mass extinction cross section Cext / mass_p [m^2/g].
    MAC : float
        Mass absorption cross section Cabs / mass_p [m^2/g].
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

    Qsca, Qext, Qabs, Qback, S11, S22 = \
        _compute_efficiencies_and_amplitudes(a, b, x, k, nstop, int(nang))

    # Cross sections and mass-normalized quantities
    geo_cs = np.pi * d_p**2 / 4.0
    mass_p = (np.pi/6.0) * (d_p*1e2)**3 * dens_p  # [g]
    MSC = geo_cs * Qsca / mass_p
    MEC = geo_cs * Qext / mass_p
    MAC = geo_cs * Qabs / mass_p

    return Qsca, Qext, Qabs, Qback, S11, S22, MSC, MEC, MAC
