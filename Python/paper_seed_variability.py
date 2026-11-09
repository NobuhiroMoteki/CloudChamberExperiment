# -*- coding: utf-8 -*-
"""SI numbers: variability of the estimates between retrainings of the same estimator (SI Sect. S7).

The 2 Nov estimator (pressure-curve library of 31 Oct-1 Nov, no pressure-sensor correction) is
retrained on the same simulations (models/sbi_shape4_nopcorr-cal1031-1101_sims_30000.npz, the last 2000
held out as in run_sbi_shape4.py) with the training seeds SEEDS. 2 Nov is then re-analyzed with each
(reanalyze.py), and the medians of N/N_CPC and qw/qw_hyg and the per-cycle ratio of N to the paper's
estimate are compared.

Usage:
  SBI4_PCORR=0 python paper_seed_variability.py train
  python reanalyze.py 2026-11-02T00:00 2026-11-03T00:00 ../data/reanalysis/reanalysis_nopcorr-cal1031-1101-seed<S>.csv \
      models/npe_shape4_nopcorr-cal1031-1101-seed<S>_30000.pt          (for each seed)
  python paper_seed_variability.py
"""
import os
import sys

import numpy as np

from paper_fig_n_validation import load
from paper_style import HERE

SEEDS = (1, 2, 3)
TAG = "nopcorr-cal1031-1101"
DAY = ("2026-11-02", "2026-11-03")


def train():
    import torch
    from run_sbi_shape import train as fit
    d = np.load(os.path.join(HERE, "models", f"sbi_shape4_{TAG}_sims_30000.npz"))
    theta, x = d["theta"][:-2000], d["x"][:-2000]
    for seed in SEEDS:
        out = os.path.join(HERE, "models", f"npe_shape4_{TAG}-seed{seed}_30000.pt")
        if not os.path.exists(out):
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.save(fit(theta, x), out)
            print("saved", out, flush=True)


def compare():
    base = load(f"reanalysis_{TAG}.csv", *DAY)
    print(f"paper estimator: N/CPC {(base.nd_cm3 / base.cpc).median():.4f}, "
          f"qw/hyg {(base.qw_gkg / base.qw_ref_td_gkg).median():.4f}")
    for s in SEEDS:
        d = load(f"reanalysis_{TAG}-seed{s}.csv", *DAY)
        m = d.merge(base[["datetime", "nd_cm3"]], on="datetime", suffixes=("", "_0"))
        r = m.nd_cm3 / m.nd_cm3_0
        print(f"seed {s}: N/CPC {(d.nd_cm3 / d.cpc).median():.4f}, qw/hyg {(d.qw_gkg / d.qw_ref_td_gkg).median():.4f}, "
              f"N/N_paper median {r.median():.4f} (IQR {r.quantile(.25):.4f}-{r.quantile(.75):.4f})")


if __name__ == "__main__":
    train() if sys.argv[1:] == ["train"] else compare()
