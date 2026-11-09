# -*- coding: utf-8 -*-
"""Physical constants (KankyoJissu2022 Appendix A, same values as atmospheric_composition.py)."""

# dry air
RD = 287.04  # [J kg-1 K-1] gas constant
CPD = 1005.0  # [J kg-1 K-1] specific heat at constant P
KA = 0.025  # [W m-1 K-1] thermal conductivity @ 0~20C

# water
EPS = 0.622  # Mv/Md
RV = 461.5  # [J kg-1 K-1] gas constant of vapor
CPV = 1850.0  # [J kg-1 K-1] specific heat of vapor at constant P
DV = 2.5e-5  # [m2 s-1] diffusivity of vapor in air @ 0~20C
CL = 4218.0  # [J kg-1 K-1] specific heat of liquid water (atmospheric_composition.py had a typo 4128)
RHO_L = 1.0e3  # [kg m-3] liquid water density
LV = 2.501e6  # [J kg-1] enthalpy of vaporization @ 0C
N_WATER = 1.33  # refractive index of liquid water @ visible
K_WATER = 2e-8  # imaginary index of liquid water @ 670 nm (approx., Hale and Querry 1973)

# saturation vapor pressure es(T) = A_ES*exp(-B_ES/T)  (Problem 2.2)
A_ES = 2.5368e11  # [Pa]
B_ES = 5.4205e3  # [K]

# chamber / optics
WAVELENGTH = 670e-9  # [m] laser wavelength
L_PATH = 0.2  # [m] optical path length (chamber diameter)
