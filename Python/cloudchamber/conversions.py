# -*- coding: utf-8 -*-
"""Sensor voltage <-> physical quantity conversions (same equations as the LabVIEW VI)."""
import numpy as np

# thermistor divider (Docs Sec. 4.2, values from the VI formula node)
T0_TH = 298.15  # [K]
B_TH = 3435.0  # [K]
VIN_TH = 5.07  # [V] supply voltage of the divider
R1_TH = 29.5e3  # [ohm] series resistor
RIN_ADC = 1.75e6  # [ohm] USB-6008 input impedance (USB-6210: >10 GOhm, see daq.DAQ_MODELS)
R0_TH = 9.2e3  # [ohm] thermistor resistance at T0 (nominal 10 kOhm; calibrated 2026-10-29 against
#                the thermocouple supplied with the DPH-203B hygrometer, with calib_thermistor.py)


def pressure_pa(v):
    """Setra 276: 0.1-5.1 V <-> 0-1380 hPa."""
    return (-27.6 + 276.0 * np.asarray(v)) * 1e2


def pressure_to_volt(p_pa):
    return (np.asarray(p_pa) * 1e-2 + 27.6) / 276.0


# chilled-mirror hygrometer DPH-203B (Tokyo Koden Kogyo): dew-point voltage output, nominal 10 mV/C
# (its manual wrongly states a 4-20 mA current output). Linear fit to the 5-point calibration
# (Td = -10.3..30.2 C, ambient 20-22 C): residual rms 0.06 C.
TD_SLOPE = 0.100574  # [C per mV]
TD_OFFSET = -0.1018  # [C]


def dewpoint_c(v):
    """Chilled-mirror dew point [degC] from its voltage output [V]."""
    return TD_SLOPE * np.asarray(v, dtype=float) * 1e3 + TD_OFFSET


def dewpoint_to_volt(td_c):
    return (np.asarray(td_c, dtype=float) - TD_OFFSET) / TD_SLOPE * 1e-3


def qw_from_dewpoint(td_c, p_pa):
    """Water-vapour mixing ratio [kg/kg] from dew point, with the WMO (2018) Magnus formula over water
    (more accurate than the two-constant Clausius-Clapeyron form used inside the forward model)."""
    td_c = np.asarray(td_c, dtype=float)
    e = 611.2 * np.exp(17.62 * td_c / (243.12 + td_c))
    return 0.622 * e / (p_pa - e)


def dewpoint_from_qw(qw, p_pa):
    """Dew point [degC] of mixing ratio qw [kg/kg] at pressure p (inverse of qw_from_dewpoint)."""
    qw = np.asarray(qw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(qw * p_pa / (0.622 + qw) / 611.2)
        return 243.12 * g / (17.62 - g)


def number_per_cm3(n_per_kg, p_pa, T_k):
    """Number concentration [#/cm3] from [#/kg] at pressure p and temperature T (dry-air density,
    the same conversion as used for the CPC comparison)."""
    return np.asarray(n_per_kg, dtype=float) * p_pa / (287.04 * T_k) * 1e-6


def hygrometer_rh(v):
    """Chilled-mirror hygrometer RH output: 0 V = 0 %, 1 V = 100 % (returned as a fraction 0-1)."""
    return np.asarray(v, dtype=float)


def rh_to_volt(rh):
    return np.asarray(rh, dtype=float)


def thermistor_k(v, r0=R0_TH, rin=RIN_ADC):
    """Temperature [K] from the divider output voltage (B-parameter equation); rin = ADC input impedance."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (VIN_TH / v - 1.0) * r0 / R1_TH - r0 / rin
        return 1.0 / (1.0 / T0_TH - np.log(arg) / B_TH)


def thermistor_r0(v, T, rin=RIN_ADC):
    """R0 [ohm] that makes thermistor_k(v, R0, rin) == T (one-point calibration)."""
    return np.exp(B_TH * (1.0 / T0_TH - 1.0 / T)) / ((VIN_TH / v - 1.0) / R1_TH - 1.0 / rin)


def thermistor_to_volt(T, r0=R0_TH, rin=RIN_ADC):
    r_th = r0 * np.exp(B_TH * (1.0 / np.asarray(T) - 1.0 / T0_TH))
    return VIN_TH / (R1_TH * (1.0 / r_th + 1.0 / rin) + 1.0)
