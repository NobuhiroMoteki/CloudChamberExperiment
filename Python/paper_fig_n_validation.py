# -*- coding: utf-8 -*-
"""Paper data: leave-one-day-out estimates and the CPC (3022A), 2026-10-31 12:18 to 2026-11-02 16:49.

Each day is estimated by the v4 posterior built without it (leave-one-day-out: run_sbi_shape4.py
--tag nopcorr-lodo1031, nopcorr-lodo1101, nopcorr-cal1031-1101 with SBI4_PCORR=0, no pressure-sensor
correction), so neither the CPC nor the hygrometer enters the estimates. The CPC value of a cycle is the median of the 1 Hz CPC over the 20 s before the valve
opens. DAYS and load() are shared by the paper scripts; the figure is made by paper_fig_validation.py.
"""
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

import paper_style as ps
from paper_style import DATA

DAYS = [(lab, f, a, b, col, mk) for (lab, f, a, b), col, mk in zip(
    [("31 Oct", "reanalysis_nopcorr-lodo1031.csv", "2026-10-31T12:18", "2026-11-01"),
     ("1 Nov", "reanalysis_nopcorr-lodo1101.csv", "2026-11-01", "2026-11-02"),
     ("2 Nov", "reanalysis_nopcorr-cal1031-1101.csv", "2026-11-02", "2026-11-03")], ps.DAY_COLORS, ps.DAY_MARKERS)]
# Marin-5 nebulising water near the inlets: short dense plumes, the flask and the CPC see different air
EXCLUDE = [("2026-11-02T10:00", "2026-11-02T10:18")]
LO, HI = 100, 15000  # [cm-3] scatter axes
TICKS = [100, 1000, 10000]
CPC_LINE = ps.REF


def cpc_1hz(day):
    c = pd.read_csv(os.path.join(DATA, "cpc", f"cpc_{day}.csv"), low_memory=False)
    f = pd.to_numeric(c.flow_cm3s, errors="coerce")
    return c.Time_stamp.to_numpy(), np.where(f > 0, c.counts_1s / f.where(f > 0, 1), np.nan)


def load(fname, start, end):
    d = pd.read_csv(os.path.join(DATA, "reanalysis", fname))
    d = d[(d.datetime >= start) & (d.datetime < end) & d.nd_cm3.notna()].copy()
    for a, b in EXCLUDE:
        d = d[(d.datetime < a) | (d.datetime >= b)]
    ct, cn = cpc_1hz(start[:10].replace("-", ""))
    d["cpc"] = [np.nanmedian(cn[(ct >= t - 20) & (ct < t)]) for t in d.Time_stamp]
    d["t"] = pd.to_datetime(d.datetime)
    return d[d.cpc > 0]


def log_ticks(axis):
    axis.set_major_locator(FixedLocator(TICKS))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
