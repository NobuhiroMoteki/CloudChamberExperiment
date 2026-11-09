# -*- coding: utf-8 -*-
"""Paper numbers: sensitivity of the estimates to errors of the measured T_ini and p_ini.

For every STEP-th cycle of the validation, the observation vector y of the leave-one-day-out
estimator of its day is evaluated as recorded and with T_ini shifted by +-DT [K] or p_ini shifted by
+-DP [hPa] (an offset of the pressure sensor; the pressure drops in y are unchanged), or with the
pressure drops in y scaled by 1 +- DS (span error and non-linearity of the sensor). The posterior
median of qw and of N in cm-3 (converted with the shifted T_ini and p_ini, as the instrument would)
is compared with that of the unshifted y; the result is the median over the cycles of the ratio.

Usage: python paper_input_sensitivity.py  -> models/input_sensitivity.csv (read by paper_table_uncertainty.py)
"""
import os

import numpy as np
import torch

from cloudchamber import conversions as cv
from cloudchamber import sbi_shape4 as ss4
from cloudchamber.sbi_shape2 import SHAPE_OFFSETS
from cloudchamber.shape_estimator import cycle_x
from fit_pressure_correction import raw_path
from paper_fig_n_validation import DAYS, load
from paper_style import HERE

MODELS = ("npe_shape4_nopcorr-lodo1031_30000.pt", "npe_shape4_nopcorr-lodo1101_30000.pt",
          "npe_shape4_nopcorr-cal1031-1101_30000.pt")
STEP = 15
DT = 0.1  # [K] accuracy of the thermocouple against which the thermistor was calibrated (one point)
DP = 3.4  # [hPa] stated accuracy of the Setra 276, 0.25 % of 1379 hPa (data sheet SS276 rev. E)
DS = 0.007  # [-] relative error of a drop of ~500 hPa from an error of 3.4 hPa (span, non-linearity)
N_SAMPLES = 2000
I_T, I_P = ss4.X_NAMES.index("T_ini"), ss4.X_NAMES.index("p_ini_hPa")
I_D = [i for i, n in enumerate(ss4.X_NAMES) if n.startswith("dp")]


def median(post, x):
    s = post.sample((N_SAMPLES,), x=torch.tensor(x, dtype=torch.float32), show_progress_bars=False).numpy()
    return np.median(s[:, 0]), np.median(s[:, 1])


def main():
    torch.manual_seed(0)
    res = {k: [] for k in ("T+", "T-", "p+", "p-", "s+", "s-")}
    for (lab, f, a, b, _, _), m in zip(DAYS, MODELS):
        post = torch.load(os.path.join(HERE, "models", m), weights_only=False)
        for row in load(f, a, b).iloc[::STEP].itertuples():
            d = np.load(raw_path(row.datetime))
            t, p, tr = d["t"].astype(float), d["p"].astype(float) * 1e2, d["tr"].astype(float)
            p_ini, T_ini = float(d["p_ini"]) * 1e2, float(d["T_ini"])
            x, _ = cycle_x(t, p, tr, p_ini, T_ini, offsets=SHAPE_OFFSETS, observed_x=ss4.observed_x)
            if x is None:
                continue
            q0, l0 = median(post, x)
            n0 = 10 ** l0 * cv.number_per_cm3(1.0, p_ini, T_ini)
            for key, dT, dp, ds in (("T+", DT, 0, 0), ("T-", -DT, 0, 0), ("p+", 0, DP, 0), ("p-", 0, -DP, 0),
                                    ("s+", 0, 0, DS), ("s-", 0, 0, -DS)):
                xx = np.array(x, dtype=float)
                xx[I_T] += dT
                xx[I_P] += dp
                xx[I_D] *= 1 + ds
                q, l = median(post, xx)
                n = 10 ** l * cv.number_per_cm3(1.0, p_ini + dp * 1e2, T_ini + dT)
                res[key].append((q / q0, n / n0))
        print(lab, "done", flush=True)
    rows = []
    for key, v in res.items():
        v = np.array(v)
        rows.append(f"{key},{np.median(v[:, 0]):.5f},{np.median(v[:, 1]):.5f}")
        print(f"{key}: n {len(v)}, qw ratio median {np.median(v[:, 0]):.4f}, N ratio median {np.median(v[:, 1]):.4f} "
              f"(IQR {np.percentile(v[:, 1], 25):.4f}-{np.percentile(v[:, 1], 75):.4f})")
    with open(os.path.join(HERE, "models", "input_sensitivity.csv"), "w") as fh:
        fh.write(f"# DT {DT} K, DP {DP} hPa, DS {DS}\nshift,qw_ratio,N_ratio\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
