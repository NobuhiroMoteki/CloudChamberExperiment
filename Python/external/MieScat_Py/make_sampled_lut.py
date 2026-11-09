"""pcas_lut_schema producer: Mie sphere sampled table (CAS-v2 forward amplitudes).

Thin CLI for the pcas_lut_schema adapter:

    python make_sampled_lut.py --preset {air,liquid} --output out.h5

Writes a contract-conformant ``lut_sampled`` HDF5 (columnar). Spheres are
isotropic, so there are no orientation columns and the forward amplitudes are
degenerate: ``S_p == S_s == S(0)`` at theta=0 (species_kind = "sphere"). Rows
enumerate a (wavelength, D_ve, Re(m_p), Im(m_p)) grid. Provenance uses the
vendored contract canonical config hash so the config_hash matches consumers.

Requires numpy + h5py in the run environment.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import pathlib
import struct
import subprocess

import h5py
import numpy as np

from mie_complex_amplitudes import mie_complex_amplitudes

# vendored from pcas_lut_schema/reference/canonical_hash.py (contract v0.1.0)
# language-independent, bit-exact byte encoding; do not edit — re-vendor from source.
def _canonical_encode(v):
    if v is None:
        return b"n;"
    if isinstance(v, bool):
        return b"b1;" if v else b"b0;"
    if isinstance(v, int):
        return b"i" + str(v).encode("ascii") + b";"
    if isinstance(v, float):
        return b"f" + struct.pack(">d", v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"s" + str(len(b)).encode("ascii") + b":" + b
    if isinstance(v, (list, tuple)):
        return b"l" + str(len(v)).encode("ascii") + b":" + b"".join(_canonical_encode(x) for x in v)
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: str(kv[0]))
        body = b"".join(_canonical_encode(str(k)) + _canonical_encode(val) for k, val in items)
        return b"d" + str(len(items)).encode("ascii") + b":" + body
    raise TypeError(f"canonical_hash: unsupported type {type(v).__name__}")


def _provenance_attrs(config):
    repo = pathlib.Path(__file__).resolve().parent

    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    return {
        "producer_repo": "MieScat_Py",
        "producer_version": "0.2.0",
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "contract_version": "0.1.0",
        "config_hash": hashlib.sha256(_canonical_encode(config)).hexdigest(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# --- presets and grids ---
_PRESET_MEDIUM = {
    "air": [(0.638, 1.0), (0.834, 1.0)],
    "liquid": [(0.637, 1.3315), (0.773, 1.3300)],  # water host at 637/773 nm
}
_D_VE_RANGE = (0.2, 0.8, 4)      # volume-equivalent DIAMETER [um]
_M_REAL_RANGE = (1.35, 1.55, 3)  # Re(m_p) (near-transparent spheres)
_M_IMAG_RANGE = (0.0, 0.0, 1)    # Im(m_p) (transparent)


def _spec(rng):
    return {"min": float(rng[0]), "max": float(rng[1]), "n": int(rng[2])}


def main():
    parser = argparse.ArgumentParser(description="Mie sphere sampled-table producer (pcas_lut_schema)")
    parser.add_argument("--preset", choices=["air", "liquid"],
                        default=os.environ.get("MIE_SWEEP_PRESET", "liquid"))
    parser.add_argument("--output", default=None)
    cli, _ = parser.parse_known_args()
    preset = cli.preset.lower()
    medium = _PRESET_MEDIUM[preset]
    out = cli.output or f"mie_sampled_{preset}.h5"

    config = {
        "preset": preset,
        "medium_conditions": [[float(wl), float(mm)] for wl, mm in medium],
        "d_ve_grid": _spec(_D_VE_RANGE),
        "m_real_grid": _spec(_M_REAL_RANGE),
        "m_imag_grid": _spec(_M_IMAG_RANGE),
    }

    d_grid = np.linspace(*_D_VE_RANGE)
    mr_grid = np.linspace(*_M_REAL_RANGE)
    mi_grid = np.linspace(*_M_IMAG_RANGE)

    cols = {k: [] for k in
            ("wl_0", "m_m", "m_real", "m_imag", "D_ve", "S_s_re", "S_s_im", "converged")}
    for wl, mm in medium:
        for D in d_grid:
            for mr in mr_grid:
                for mi in mi_grid:
                    s_re, s_im, _, _ = mie_complex_amplitudes(wl, mm, float(D), float(mr), float(mi))
                    cols["wl_0"].append(wl)
                    cols["m_m"].append(mm)
                    cols["m_real"].append(float(mr))
                    cols["m_imag"].append(float(mi))
                    cols["D_ve"].append(float(D))
                    cols["S_s_re"].append(float(s_re))
                    cols["S_s_im"].append(float(s_im))
                    cols["converged"].append(1)

    n = len(cols["wl_0"])
    with h5py.File(out, "w") as f:
        f.attrs["species_kind"] = "sphere"
        f.attrs["sampling"] = "grid(wl,D_ve,m_real,m_imag)"
        f.create_dataset("row_id", data=np.arange(n, dtype=np.int64))
        for name in ("wl_0", "m_m", "m_real", "m_imag", "D_ve", "S_s_re", "S_s_im"):
            f.create_dataset(name, data=np.asarray(cols[name], dtype=np.float64))
        # spheres: S_p == S_s (degenerate at theta=0)
        f.create_dataset("S_p_re", data=np.asarray(cols["S_s_re"], dtype=np.float64))
        f.create_dataset("S_p_im", data=np.asarray(cols["S_s_im"], dtype=np.float64))
        f.create_dataset("converged", data=np.asarray(cols["converged"], dtype=np.int8))
        pg = f.create_group("provenance")
        for k, v in _provenance_attrs(config).items():
            pg.attrs[k] = v

    print(f"Wrote {out}  ({n} rows, preset={preset})")


if __name__ == "__main__":
    main()
