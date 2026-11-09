# -*- coding: utf-8 -*-
"""Paper numbers: the gain-lag valley fitted without each day (leave-one-day-out, SI Sect. S4).

Superseded 2026-11-08: the paper no longer uses the pressure-sensor correction (sbi_shape4.PCORR).

The grids come from fit_pressure_correction.py (alpha_c = 0.3, 400 cycles) on the cycles of the
evaluation period with one day removed, e.g.
  python fit_pressure_correction.py without_2026-10-31.csv --n 400 --alpha_c 0.3 \
      --out models/pressure_correction_grid_fwd_ac0.3_without1031.npz
where without_*.csv is reanalysis_cal1031-1101.csv from 2026-10-31T12:18 without that day. For each grid
the summed cost (as in paper_si_valley.py) gives the best lag at each gain (parabolic refinement on
the lag grid); a line through the gains 0.90-1.02 gives the valley lag at gain P_GAIN_REF and its slope.

Usage: python paper_valley_lodo.py
"""
import os

import numpy as np

from cloudchamber import sbi_shape4 as ss4
from paper_style import HERE

GRIDS = [("all days", "pressure_correction_grid_fwd_ac0.3.npz"),
         ("without 31 Oct", "pressure_correction_grid_fwd_ac0.3_without1031.npz"),
         ("without 1 Nov", "pressure_correction_grid_fwd_ac0.3_without1101.npz"),
         ("without 2 Nov", "pressure_correction_grid_fwd_ac0.3_without1102.npz")]


def valley(path):
    """Valley lag [s] at P_GAIN_REF, its slope [s] and the best lag at each gain."""
    d = np.load(path)
    c = (d["dts"] / 0.02) ** 2 + (d["rms"] / 0.003) ** 2
    tot = np.where(np.isfinite(c), c, 1e6).min(axis=2).sum(0)  # shape: (gain, lag)
    gains, lags = d["gains"], d["lags"]
    best = np.empty(gains.size)
    for i in range(gains.size):
        k = min(max(int(tot[i].argmin()), 1), lags.size - 2)
        y0, y1, y2 = tot[i, k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        best[i] = lags[k] + (0.5 * (y0 - y2) / den * (lags[1] - lags[0]) if den > 0 else 0.0)
    sel = (gains >= 0.90) & (gains <= 1.02)
    slope, lag_ref = np.polyfit(gains[sel] - ss4.P_GAIN_REF, best[sel], 1)
    return lag_ref, slope, gains, best


def main():
    prior = lambda g: ss4.P_LAG_REF + ss4.LAG_SLOPE * (g - ss4.P_GAIN_REF)  # noqa: E731
    for name, f in GRIDS:
        lag_ref, slope, gains, best = valley(os.path.join(HERE, "models", f))
        g = np.array(ss4.P_GAIN)
        off = lag_ref + slope * (g - ss4.P_GAIN_REF) - prior(g)  # valley minus prior centre at the gain edges
        print(f"{name:15s} valley lag at gain {ss4.P_GAIN_REF}: {lag_ref:.3f} s, slope {slope:.2f} s; "
              f"offset from the prior centre at gain {g[0]}/{g[1]}: {off[0]:+.3f}/{off[1]:+.3f} s "
              f"(prior half width {ss4.LAG_HALF} s)")


if __name__ == "__main__":
    main()
