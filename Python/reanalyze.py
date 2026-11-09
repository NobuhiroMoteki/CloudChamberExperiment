# -*- coding: utf-8 -*-
"""Re-analyse recorded decompressions (data/raw_YYYYMMDD/decomp_*.npz) with the current analysis and
SBI estimator, e.g. after a software fix or a new model.

Usage: python reanalyze.py START END [out.csv] [model.pt]
  START, END: local times, e.g. 2026-10-31T12:18 2026-11-02T14:17
Writes one row per cycle (default data/reanalysis/reanalysis_<model>.csv); pressures in hPa.
"""
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from cloudchamber import conversions as cv
from cloudchamber.analysis import analyze_decompression
from cloudchamber.shape_estimator import ShapeSBIEstimator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
I0_MIN = 1.0  # [V], as in controller.py


def raw_files(start, end):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "raw_*", "decomp_*.npz"))):
        stamp = datetime.strptime(os.path.basename(f)[7:22], "%Y%m%d_%H%M%S")
        if start <= stamp <= end:
            out.append((stamp, f))
    return out


def main():
    start, end = (datetime.fromisoformat(a) for a in sys.argv[1:3])
    est = ShapeSBIEstimator(path=sys.argv[4] if len(sys.argv) > 4 else None)
    name = os.path.splitext(os.path.basename(est.path))[0]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA, "reanalysis", f"reanalysis_{name}.csv")
    files = raw_files(start, end)
    print(f"{len(files)} cycles, model {est.path}", flush=True)
    rows, t_start = [], time.time()
    for k, (stamp, f) in enumerate(files):
        d = np.load(f)
        c = dict(t=d["t"].astype(float), p=d["p"].astype(float) * 1e2, tr=d["tr"].astype(float),
                 p_ini=float(d["p_ini"]) * 1e2, T_ini=float(d["T_ini"]), I0=float(d["I0"]))
        res = analyze_decompression(c["t"], c["p"], c["tr"], c["p_ini"])
        flags = list(res.pop("flags"))
        r = dict(Time_stamp=float(d["t_wall"]), datetime=stamp.isoformat(), T_ini=c["T_ini"],
                 p_ini=c["p_ini"] * 1e-2, I0=c["I0"], ref_Td=float(d["ref_Td"]),
                 qw_ref_td_gkg=cv.qw_from_dewpoint(float(d["ref_Td"]), c["p_ini"]) * 1e3, **res)
        if not c["I0"] >= I0_MIN:
            flags.append("no_laser")
        else:
            e = est(c)
            reason = e.pop("sbi_reason", None)
            if reason:
                flags.append(reason)
            r.update(e)
        for s in ("", "_lo", "_hi"):
            r[f"nd_cm3{s}"] = float(cv.number_per_cm3(10 ** r.get(f"log10nd{s}", np.nan), c["p_ini"], c["T_ini"]))
        r["flags"] = "|".join(flags)
        rows.append(r)
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(files)} ({time.time()-t_start:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    for col in ("p_offset", "p_c", "p_offset2"):
        if col in df:
            df[col] *= 1e-2
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    ok = df.qw_gkg.notna()
    print(f"saved {out}: {len(df)} cycles, estimated {ok.sum()}, flags "
          f"{df['flags'].fillna('').replace('', 'none').value_counts().to_dict()}")


if __name__ == "__main__":
    main()
