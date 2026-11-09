# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:00:56 2022

@author: Moteki
"""

import numpy as np
from _mie_core import _compute_bessel_and_dd, _compute_efficiencies_and_amplitudes

# Physical constants (SI units)
_E_CHARGE = 1.602176634e-19   # Elementary charge [C]
_EPS_0    = 8.8541878188e-12  # Permittivity of free space [F/m]
_MASS_E   = 9.1093837139e-31  # Electron mass [kg]
_C_LIGHT  = 2.99792458e8      # Speed of light [m/s]
_H_BAR    = 6.62607015e-34 / (2.0*np.pi)  # Reduced Planck constant [J s]
_K_B      = 1.380649e-23      # Boltzmann constant [J/K]
_T_K      = 298.0             # Reference temperature [K]


def miescat_charged(wl_0, d_p, m_p_real, m_p_imag, n_e=0, gamma_fac=1.0,
                    dens_p=1.0, nang=3):
    """Mie scattering properties of a charged single homogeneous sphere in vacuum.

    Calculates light scattering properties based on the formulations of
    Klacka and Kocifaj (2010) [KK10] for a sphere carrying n_e elementary
    charges in vacuum.

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
    d_p : float
        Particle diameter [m].
    m_p_real : float
        Real part of the particle refractive index.
    m_p_imag : float
        Imaginary part of the particle refractive index.
    n_e : int or float, optional
        Number of elementary charges on the particle. Default 0.
    gamma_fac : float, optional
        Dimensionless correction factor for gamma_s. Default 1.0.
    dens_p : float, optional
        Particle density [g/cm^3]. Default 1.0.
    nang : int, optional
        Number of scattering angle grid points (0 to 180 deg). Default 3.

    Returns
    -------
    surf_potential : float
        Electrostatic surface potential [V].
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
    r_p = d_p / 2.0
    k   = 2.0*np.pi / wl_0       # wavenumber in vacuum
    w   = k * _C_LIGHT
    x   = k * r_p

    # Surface potential and G-parameter (KK10, Eq.26)
    surf_potential = n_e * _E_CHARGE / (4.0*np.pi*_EPS_0*r_p)
    gamma_s = gamma_fac * _K_B * _T_K / _H_BAR  # KK10, Eq.37
    gamma_over_w = gamma_s / w
    g = (_E_CHARGE * surf_potential) / (_MASS_E * _C_LIGHT**2) \
        * (1.0/x) * (1.0 + 1j*gamma_over_w) / (1.0 + gamma_over_w**2)

    nstop = int(np.floor(abs(x) + 4.0*abs(x)**0.3333 + 2))

    # m_r = m_p for vacuum medium (m_m = 1)
    DD, psi, xi = _compute_bessel_and_dd(x, m_p, nstop)

    # Partial wave coefficients a_n, b_n for a charged sphere (KK10)
    n_arr = np.arange(1, nstop + 1, dtype=float)
    gD_over_mp = g * DD[1:] / m_p        # g * D_n / m_p
    n_over_x   = n_arr / x

    a = np.zeros(nstop + 1, dtype=complex)
    b = np.zeros(nstop + 1, dtype=complex)
    num_a = ((1.0 + n_arr*g/x)*DD[1:]/m_p + n_over_x)*psi[1:] \
            - (1.0 + gD_over_mp)*psi[:-1]
    den_a = ((1.0 + n_arr*g/x)*DD[1:]/m_p + n_over_x)*xi[1:]  \
            - (1.0 + gD_over_mp)*xi[:-1]
    a[1:] = num_a / den_a

    num_b = (m_p*DD[1:] + n_over_x*(1.0 - g*x/n_arr))*psi[1:] - psi[:-1]
    den_b = (m_p*DD[1:] + n_over_x*(1.0 - g*x/n_arr))*xi[1:]  - xi[:-1]
    b[1:] = num_b / den_b

    Qsca, Qext, Qabs, Qback, S11, S22 = \
        _compute_efficiencies_and_amplitudes(a, b, x, k, nstop, int(nang))

    # Cross sections and mass-normalized quantities
    geo_cs = np.pi * d_p**2 / 4.0
    mass_p = (np.pi/6.0) * (d_p*1e2)**3 * dens_p  # [g]
    MSC = geo_cs * Qsca / mass_p
    MEC = geo_cs * Qext / mass_p
    MAC = geo_cs * Qabs / mass_p

    return surf_potential, Qsca, Qext, Qabs, Qback, S11, S22, MSC, MEC, MAC
