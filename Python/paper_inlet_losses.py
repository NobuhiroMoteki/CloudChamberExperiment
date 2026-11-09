# -*- coding: utf-8 -*-
"""Paper numbers: diffusion losses in the inlets of the chamber and of the CPC (paper Sect. 7.1, SI Sect. S11).

Chamber: 1.16 m of 4.8 mm i.d. tubing (inlet-valve 1 0.50 m, valve 1-flask 0.66 m) at the flushing flow of 30-40 L/min (turbulent; deposition
velocity of Hinds 1999, 2nd ed., Eqs. 7.31-7.32, written with the Schmidt number). CPC (TSI 3022A): 1.2 m of 5 mm i.d. conductive silicone
tubing at 0.30 L/min (laminar; Gormley-Kennedy as given by Hinds 1999, 2nd ed., Eq. 7.29; the third term
0.0325 exp(-179 mu) of the original, which Hinds omits, changes the losses by at most 0.3 % points). Size-averaged losses
are number-weighted over lognormal dry diameters above a sharp 7 nm cut-off, with the count median
diameter (2 r_g) and sigma_g spanning the priors of the simulator.

Usage: python paper_inlet_losses.py
"""
import numpy as np
from scipy.constants import k as K_B, pi

T, P_AIR = 296.0, 990e2  # [K], [Pa] typical room air of the evaluation
D_TUBE = 0.19 * 0.0254  # [m] inner diameter of both inlets (TSI 3001788 conductive tubing, 0.19 in)
CHAMBER = (1.16, 30e-3 / 60, 40e-3 / 60)  # length [m], flushing flow range [m^3/s]
CPC = (1.2, 0.30e-3 / 60)  # length [m], flow [m^3/s]
CUTOFF = 7e-9  # [m] nominal D50 of the TSI 3022A
CMD = (40e-9, 160e-9)  # [m] count median diameter, 2 x prior range of r_g (20-80 nm)
SIGMA_G = (1.4, 2.2)  # prior range of sigma_g

S_SUTH = 110.4  # [K] Sutherland constant of air
MU_AIR = 1.81e-5 * (T / 293.15) ** 1.5 * (293.15 + S_SUTH) / (T + S_SUTH)  # [Pa s]
LAMBDA_AIR = 66.5e-9 * (101325 / P_AIR) * (T / 293.15) * (1 + S_SUTH / 293.15) / (1 + S_SUTH / T)  # [m]
RHO_AIR = P_AIR / (287.05 * T)  # [kg/m^3]


def diffusivity(dp):
    """Particle diffusion coefficient [m^2/s] for diameter dp [m] (Cunningham-corrected)."""
    kn = 2 * LAMBDA_AIR / dp
    cc = 1 + kn * (1.142 + 0.558 * np.exp(-0.999 / kn))
    return K_B * T * cc / (3 * pi * MU_AIR * dp)


def reynolds(q, d=D_TUBE):
    return RHO_AIR * 4 * q / (pi * d * MU_AIR)


def penetration_laminar(dp, length, q):
    """Gormley-Kennedy penetration of a laminar tube flow q [m^3/s] over length [m]."""
    m = diffusivity(dp) * length / q
    return np.where(m < 0.009, 1 - 5.50 * m ** (2 / 3) + 3.77 * m,
                    0.819 * np.exp(-11.5 * m) + 0.0975 * np.exp(-70.1 * m))


def penetration_turbulent(dp, length, q, d=D_TUBE):
    """Penetration of a turbulent tube flow, deposition velocity 0.04 U Re^-1/4 Sc^-2/3."""
    u = q / (pi * d ** 2 / 4)
    sc = MU_AIR / (RHO_AIR * diffusivity(dp))
    v_dep = 0.04 * u / (reynolds(q, d) ** 0.25 * sc ** (2 / 3))
    return np.exp(-4 * v_dep * length / (d * u))


def size_averaged_loss(penetration, cmd, sigma_g):
    """Number fraction lost above CUTOFF for a lognormal of count median diameter cmd [m]."""
    dp = np.logspace(np.log10(CUTOFF), np.log10(3e-6), 4000)  # shape: (n,)
    n = np.exp(-0.5 * (np.log(dp / cmd) / np.log(sigma_g)) ** 2) / dp
    return 1 - np.trapezoid(n * penetration(dp), dp) / np.trapezoid(n, dp)


def ratio_factor_range():
    """Range of (1 - chamber loss) / (1 - CPC loss) over the prior size distributions."""
    q_mid = 0.5 * (CHAMBER[1] + CHAMBER[2])
    f = [(1 - size_averaged_loss(lambda d: penetration_turbulent(d, CHAMBER[0], q_mid), c, s))
         / (1 - size_averaged_loss(lambda d: penetration_laminar(d, *CPC), c, s))
         for c in CMD for s in SIGMA_G]
    return min(f), max(f)


def main():
    dps = np.array([3, 5, 7, 10, 20, 30]) * 1e-9  # shape: (6,)
    for q in CHAMBER[1:]:
        loss = 1 - penetration_turbulent(dps, CHAMBER[0], q)
        print(f"chamber {q * 6e4:.0f} L/min: Re {reynolds(q):.0f}, residence "
              f"{CHAMBER[0] * pi * D_TUBE ** 2 / 4 / q:.2f} s, loss %",
              " ".join(f"{d * 1e9:.0f}nm {100 * x:.1f}" for d, x in zip(dps, loss)))
    loss = 1 - penetration_laminar(dps, *CPC)
    print(f"CPC {CPC[1] * 6e4:.2f} L/min: Re {reynolds(CPC[1]):.0f}, residence "
          f"{CPC[0] * pi * D_TUBE ** 2 / 4 / CPC[1]:.2f} s, loss %",
          " ".join(f"{d * 1e9:.0f}nm {100 * x:.1f}" for d, x in zip(dps, loss)))
    q_mid = 0.5 * (CHAMBER[1] + CHAMBER[2])
    for c in CMD:
        for s in SIGMA_G:
            lc = size_averaged_loss(lambda d: penetration_turbulent(d, CHAMBER[0], q_mid), c, s)
            lp = size_averaged_loss(lambda d: penetration_laminar(d, *CPC), c, s)
            print(f"CMD {c * 1e9:.0f} nm, sigma_g {s}: loss above 7 nm chamber {100 * lc:.1f} %, "
                  f"CPC {100 * lp:.1f} %, N/N_CPC x {(1 - lc) / (1 - lp):.3f}")
    print("N/N_CPC factor range: %.3f-%.3f" % ratio_factor_range())


if __name__ == "__main__":
    main()
