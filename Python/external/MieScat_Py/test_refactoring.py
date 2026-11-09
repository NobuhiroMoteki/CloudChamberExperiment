"""
Numerical validation: compare original vs refactored Mie scattering modules.

Original modules are imported from _original/ subdirectory.
Refactored modules are imported from the project root.
All outputs must agree to within rtol=1e-12.
"""

import sys
import numpy as np

# ── import originals from backup ──────────────────────────────────────────────
sys.path.insert(0, "_original")
from miescat             import miescat             as miescat_old
from miescat_charged     import miescat_charged     as miescat_charged_old
from mie_complex_amplitudes import mie_complex_amplitudes as mca_old
sys.path.pop(0)

# ── import refactored versions ────────────────────────────────────────────────
from miescat             import miescat             as miescat_new
from miescat_charged     import miescat_charged     as miescat_charged_new
from mie_complex_amplitudes import mie_complex_amplitudes as mca_new

RTOL = 1e-12
ATOL = 1e-15
NANG = 5  # use 5 angles for a richer check

pass_count = 0
fail_count = 0


def check(label, old, new):
    global pass_count, fail_count
    old = np.atleast_1d(np.asarray(old))
    new = np.atleast_1d(np.asarray(new))
    if np.allclose(old, new, rtol=RTOL, atol=ATOL):
        print(f"  PASS  {label}")
        pass_count += 1
    else:
        diff = np.max(np.abs(old - new) / (np.abs(old) + ATOL))
        print(f"  FAIL  {label}  max_rel_diff={diff:.3e}")
        print(f"         old={old}")
        print(f"         new={new}")
        fail_count += 1


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("miescat — neutral sphere")
print("=" * 60)

cases_miescat = [
    # (wl_0, m_m, d_p, m_p_real, m_p_imag, dens_p)  — from call_mie_scattering.ipynb
    dict(wl_0=0.83e-6,   m_m=1.0,     d_p=0.4e-6,  m_p_real=1.5,   m_p_imag=0.0,  dens_p=2.7),
    dict(wl_0=0.6328e-6, m_m=1.33154, d_p=0.803e-6, m_p_real=1.585, m_p_imag=0.0,  dens_p=1.0),
    # absorbing particle
    dict(wl_0=0.5e-6,    m_m=1.0,     d_p=1.0e-6,  m_p_real=1.5,   m_p_imag=0.5,  dens_p=2.0),
    # large particle
    dict(wl_0=0.5e-6,    m_m=1.33,    d_p=5.0e-6,  m_p_real=1.59,  m_p_imag=0.01, dens_p=1.1),
    # highly absorbing (soot-like)
    dict(wl_0=0.55e-6,   m_m=1.0,     d_p=0.15e-6, m_p_real=1.75,  m_p_imag=0.44, dens_p=1.8),
]

for i, kw in enumerate(cases_miescat):
    print(f"\nCase {i+1}: {kw}")
    o = miescat_old(**kw, nang=NANG)
    n = miescat_new(**kw, nang=NANG)
    labels = ["Qsca", "Qext", "Qabs", "Qback", "S11", "S22", "MSC", "MEC", "MAC"]
    for lbl, ov, nv in zip(labels, o, n):
        check(lbl, ov, nv)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("miescat_charged — charged sphere")
print("=" * 60)

cases_charged = [
    # neutral (n_e=0) — from call_mie_scattering.ipynb
    dict(wl_0=0.834e-6, d_p=10e-9, m_p_real=1.5, m_p_imag=0.0, n_e=0,   gamma_fac=0.1, dens_p=2.7),
    # charged (n_e=100) — from call_mie_scattering.ipynb
    dict(wl_0=0.834e-6, d_p=10e-9, m_p_real=1.5, m_p_imag=0.0, n_e=100, gamma_fac=0.1, dens_p=2.7),
    # different charge and size
    dict(wl_0=0.532e-6, d_p=50e-9, m_p_real=1.6, m_p_imag=0.1, n_e=50,  gamma_fac=0.5, dens_p=2.0),
    # large gamma_fac
    dict(wl_0=0.800e-6, d_p=20e-9, m_p_real=1.4, m_p_imag=0.0, n_e=200, gamma_fac=2.0, dens_p=1.5),
]

for i, kw in enumerate(cases_charged):
    print(f"\nCase {i+1}: {kw}")
    o = miescat_charged_old(**kw, nang=NANG)
    n = miescat_charged_new(**kw, nang=NANG)
    labels = ["surf_potential", "Qsca", "Qext", "Qabs", "Qback", "S11", "S22", "MSC", "MEC", "MAC"]
    for lbl, ov, nv in zip(labels, o, n):
        check(lbl, ov, nv)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("mie_complex_amplitudes — CAS amplitudes")
print("=" * 60)

cases_mca = [
    # from call_mie_scattering.ipynb
    dict(wl_0=0.6328, m_m=1.33154, d_p=0.803,  m_p_real=1.585, m_p_imag=0.0),
    # absorbing Au-like
    dict(wl_0=0.834,  m_m=1.329,   d_p=0.2,    m_p_real=0.1618, m_p_imag=5.189),
    # small particle
    dict(wl_0=0.532,  m_m=1.0,     d_p=0.05,   m_p_real=1.5,   m_p_imag=0.0),
    # large particle
    dict(wl_0=0.650,  m_m=1.33,    d_p=2.0,    m_p_real=1.59,  m_p_imag=0.02),
]

for i, kw in enumerate(cases_mca):
    print(f"\nCase {i+1}: {kw}")
    o = mca_old(**kw, nang=NANG)
    n = mca_new(**kw, nang=NANG)
    labels = ["S_fwd_real", "S_fwd_imag", "S_bak_real", "S_bak_imag"]
    for lbl, ov, nv in zip(labels, o, n):
        check(lbl, ov, nv)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Results: {pass_count} PASS, {fail_count} FAIL")
print("=" * 60)
if fail_count == 0:
    print("All tests passed — refactoring is numerically equivalent.")
else:
    sys.exit(1)
