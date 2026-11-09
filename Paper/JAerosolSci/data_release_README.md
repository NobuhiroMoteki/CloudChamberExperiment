# Data for "Working-fluid-free measurement of particle number concentration and water vapor from the laser-extinction curve of an expansion cloud by simulation-based inference"

N. Moteki, Journal of Aerosol Science (submitted). Code: https://github.com/NobuhiroMoteki/CloudChamberExperiment

Recorded data of the evaluation period, 31 Oct 2026 12:18 to 2 Nov 2026 16:49 (local time, UTC+9): 3132 cycles of 60 s
of the expansion chamber in a laboratory room, with a CPC (TSI 3022A) and a chilled-mirror hygrometer (DPH-203B,
Tokyo Opto-Electronics) sampling the same room air through their own inlets. All times in the files are UNIX times
(`Time_stamp`, s) and ISO 8601 local times (`datetime`).

## Contents

| Folder | Content |
|---|---|
| `raw/raw_YYYYMMDD/decomp_YYYYMMDD_HHMMSS.npz` | Signals of the expansion phase of every cycle (one file per cycle) |
| `cycles/cloud_data_YYYYMMDD.csv` | One row per cycle as written during the measurement (online estimates of an earlier estimator version) |
| `cpc/cpc_YYYYMMDD.csv` | CPC record at 1 Hz |
| `estimates/reanalysis_*.csv` | Estimates used in the paper, obtained by re-analyzing the raw files |

### `raw/*.npz` (NumPy compressed; `np.load(file)`)

| Key | Content | Unit |
|---|---|---|
| `t` | Time from the start of the expansion phase (1000 samples at 100 Hz) | s |
| `p` | Pressure (barometric sensor) | hPa |
| `tr` | Transmittance I/I0 | – |
| `T` | Thermistor temperature (too slow to follow the expansion) | K |
| `t_wall` | UNIX time of the start of the expansion phase | s |
| `I0` | Reference photodiode signal (median of the sealed phase) | V |
| `T_ini`, `p_ini` | Temperature and pressure before the expansion | K, hPa |
| `ref_RH`, `ref_Td` | Hygrometer relative humidity (0–1) and dew point | –, °C |
| `r0`, `rin` | Thermistor R0 and ADC input resistance used for the temperature | Ω |

### `estimates/reanalysis_*.csv`

| File | Estimator (days of its pressure-curve library) | Used in the paper for |
|---|---|---|
| `reanalysis_nopcorr-lodo1031.csv` | trained on 1–2 Nov | 31 Oct (leave-one-day-out) |
| `reanalysis_nopcorr-lodo1101.csv` | trained on 31 Oct and 2 Nov | 1 Nov (leave-one-day-out) |
| `reanalysis_nopcorr-cal1031-1101.csv` | trained on 31 Oct – 1 Nov | 2 Nov (leave-one-day-out) |
| `reanalysis_nopcorr-all.csv` | trained on all days | comparison of estimators (SI) |
| `reanalysis_nopcorr-abl-nofwd.csv` | as nopcorr-cal1031-1101, without the forward-scattering correction | Table 3 |

No estimator uses a correction of the pressure-sensor response or data of the reference instruments.

Main columns: `qw_gkg`, `qw_lo`, `qw_hi` (water vapor mixing ratio, median and 5–95 % interval, g/kg);
`log10nd`, `log10nd_lo`, `log10nd_hi` (log10 of N per kg of dry air); `nd_cm3`, `nd_cm3_lo`, `nd_cm3_hi`
(N per cm³ at `p_ini`, `T_ini`); `qw_ref_td_gkg` (hygrometer mixing ratio from `ref_Td` and `p_ini`, g/kg);
`T_ini` (K), `p_ini` (hPa), `I0` (V); `t0_ramp2`, `p_offset2`, `tau_vac1`, `frac1`, `tau_vac2`, `t_open`, `p_rms`
(fit of the pressure curve, Eq. 1 of the paper; s, hPa, –); `t_s_shape` (onset of the extinction, s after t0);
`flags` (empty if no problem).

### `cpc/cpc_YYYYMMDD.csv`

`conc_cm3` (concentration shown by the CPC), `counts_1s` (counts in 1 s), `flow_cm3s` (sample flow reported by the
CPC, cm³/s). The paper uses `counts_1s / flow_cm3s`, the median over the 20 s before each expansion.

## Notes

- Cycles from 2 Nov 10:00 to 10:18 (an aerosol-generation test near the inlets) are included but excluded from the
  comparison in the paper.
- The columns of `cycles/cloud_data_*.csv` are described in `Python/MANUAL.md` (Sect. 9) of the code repository.
  Their `qw_gkg` and `nd_cm3` are the online values of an earlier estimator, not those of the paper.
