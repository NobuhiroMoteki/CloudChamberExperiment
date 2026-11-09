# -*- coding: utf-8 -*-
"""Time the per-cycle processing of recorded cycles as in operation: feature extraction and pressure
fit (analysis.analyze_decompression) and the posterior (ShapeSBIEstimator: observation vector,
1000 posterior samples). Prints the machine, the median and 95th percentile per cycle.

Usage: python paper_timing.py [n_cycles] [model.pt]
"""
import glob
import os
import platform
import sys
import time

import numpy as np
import torch

from cloudchamber.analysis import analyze_decompression
from cloudchamber.shape_estimator import ShapeSBIEstimator
from paper_style import DATA


def machine():
    cpu = platform.processor()
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name; "
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"],
                             capture_output=True, text=True, timeout=30).stdout.split("\n")
        cpu, ram = out[0].strip(), out[1].strip()
    except Exception:  # noqa: BLE001
        ram = "?"
    return (f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} logical CPUs, {ram} GB RAM, "
            f"Python {platform.python_version()}, torch {torch.__version__} (CPU, {torch.get_num_threads()} threads)")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    est = ShapeSBIEstimator(path=sys.argv[2] if len(sys.argv) > 2 else None)
    files = sorted(glob.glob(os.path.join(DATA, "raw_20261102", "decomp_*.npz")))
    files = [files[i] for i in np.linspace(0, len(files) - 1, n + 1).round().astype(int)]
    print(machine())
    print(f"model {est.path}")
    t_ana, t_est = [], []
    for k, f in enumerate(files):
        d = np.load(f)
        c = dict(t=d["t"].astype(float), p=d["p"].astype(float) * 1e2, tr=d["tr"].astype(float),
                 p_ini=float(d["p_ini"]) * 1e2, T_ini=float(d["T_ini"]), I0=float(d["I0"]))
        t0 = time.perf_counter()
        analyze_decompression(c["t"], c["p"], c["tr"], c["p_ini"])
        t1 = time.perf_counter()
        est(c)
        t2 = time.perf_counter()
        if k > 0:  # the first cycle includes compilation and loading
            t_ana.append(t1 - t0)
            t_est.append(t2 - t1)
    a, e = np.array(t_ana), np.array(t_est)
    tot = a + e
    print(f"{len(tot)} cycles: analysis median {np.median(a):.3f} s, estimator median {np.median(e):.3f} s, "
          f"total median {np.median(tot):.3f} s, 95th percentile {np.percentile(tot, 95):.3f} s, max {tot.max():.3f} s")


if __name__ == "__main__":
    main()
