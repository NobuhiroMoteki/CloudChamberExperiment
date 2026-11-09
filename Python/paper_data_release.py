# -*- coding: utf-8 -*-
"""Data archive of the paper: the recorded data of the evaluation period as one zip for a GitHub release.

Contents (31 Oct 2026 12:18 - 2 Nov 2026 16:49): raw/ (npz of every expansion), cycles/ (cloud_data_*.csv),
cpc/ (cpc_*.csv), estimates/ (the reanalysis_*.csv used in the paper) and README.md
(Paper/JAerosolSci/data_release_README.md). Rows of the CSV files before START are dropped.
Re-run after any re-analysis, then replace the asset of the draft release data-jas2026:
  gh release upload data-jas2026 CloudChamber_JAS2026_data.zip --clobber

Usage: python paper_data_release.py OUT_DIR  -> OUT_DIR/CloudChamber_JAS2026_data.zip
"""
import csv
import os
import shutil
import sys

from paper_style import DATA, HERE

NAME = "CloudChamber_JAS2026_data"
START = "2026-10-31T12:18"
DAYS = ("20261031", "20261101", "20261102")
ESTIMATES = ("nopcorr-lodo1031", "nopcorr-lodo1101", "nopcorr-cal1031-1101", "nopcorr-all",
             "nopcorr-abl-nofwd")
README = os.path.join(HERE, os.pardir, "Paper", "JAerosolSci", "data_release_README.md")


def copy_rows(src, dst):
    """Copy a CSV, keeping the header and the rows from START on."""
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        r, w = csv.reader(fi), csv.writer(fo)
        head = next(r)
        w.writerow(head)
        i = head.index("datetime")
        w.writerows(row for row in r if row[i][:16] >= START)


def main():
    out = os.path.abspath(sys.argv[1])
    root = os.path.join(out, NAME)
    if os.path.exists(root):
        sys.exit(f"{root} exists; remove it first")
    for sub in ("raw", "cycles", "cpc", "estimates"):
        os.makedirs(os.path.join(root, sub))
    first = "decomp_" + START[:10].replace("-", "") + "_" + START[11:].replace(":", "") + "00"
    n = 0
    for day in DAYS:
        os.makedirs(os.path.join(root, "raw", f"raw_{day}"))
        for f in sorted(os.listdir(os.path.join(DATA, f"raw_{day}"))):
            if f.endswith(".npz") and f >= first:
                shutil.copy2(os.path.join(DATA, f"raw_{day}", f), os.path.join(root, "raw", f"raw_{day}", f))
                n += 1
        copy_rows(os.path.join(DATA, f"cloud_data_{day}.csv"), os.path.join(root, "cycles", f"cloud_data_{day}.csv"))
        copy_rows(os.path.join(DATA, "cpc", f"cpc_{day}.csv"), os.path.join(root, "cpc", f"cpc_{day}.csv"))
    for e in ESTIMATES:
        shutil.copy2(os.path.join(DATA, "reanalysis", f"reanalysis_{e}.csv"), os.path.join(root, "estimates"))
    shutil.copy2(README, os.path.join(root, "README.md"))
    zip_path = shutil.make_archive(os.path.join(out, NAME), "zip", out, NAME)
    print(f"{n} cycles -> {zip_path}")


if __name__ == "__main__":
    main()
