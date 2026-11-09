# MieScat_Py

[![GitHub release](https://img.shields.io/github/v/release/NobuhiroMoteki/MieScat_Py?sort=semver)](https://github.com/NobuhiroMoteki/MieScat_Py/releases/latest)

## Description

Python implementation of the Lorenz-Mie theory for calculating the scattering and absorption of plane electromagnetic waves by an isotropic homogeneous sphere.
Supports neutral spheres, electrically charged spheres, and Complex Amplitude Sensing (CAS) observables.
For details on the underlying theory and algorithms, see [docs/theory_note.pdf](docs/theory_note.pdf).

### File structure

| File | Description |
|------|-------------|
| `_mie_core.py` | Internal helper module shared by all three calculation modules. Contains vectorized implementations of Ricatti-Bessel functions, logarithmic derivatives, optical efficiency factors, and scattering amplitudes. |
| `miescat.py` | Computes optical efficiency factors (Qsca, Qext, Qabs, Qback), scattering amplitudes (S11, S22), and mass-normalized cross sections (MSC, MEC, MAC) for a neutral sphere. |
| `miescat_charged.py` | Extended version of `miescat.py` for an electrically charged sphere (Klacka & Kocifaj 2010). |
| `mie_complex_amplitudes.py` | Computes complex forward and backward scattering amplitudes (S_fwd, S_bak) observable by the Complex Amplitude Sensing (CAS-v1 and CAS-v2) protocol. |
| `call_mie_scattering.ipynb` | Demonstrates how to call each module with example parameters. |
| `make_mie_S_table.ipynb` | Generates an S-parameter lookup table over a grid of (d_p, m_p_real, m_p_imag) and saves it in parquet (pandas DataFrame) format. |
| `test_refactoring.py` | Numerical validation script. Compares the refactored modules against the originals (backed up in `_original/`) across 13 test cases using `np.allclose(rtol=1e-12)`. |

---

## Installation

Developed and tested with **Python 3.13.12** on Linux (WSL2, Ubuntu).
Also tested with Python 3.12.8 on Windows 11.

#### 1. Clone the repository

```sh
git clone https://github.com/NobuhiroMoteki/MieScat_Py.git
cd MieScat_Py
```

#### 2. Install dependencies

Using [uv](https://docs.astral.sh/uv/) (recommended):

```sh
uv venv
uv pip install -r requirements.txt
```

Or using pip:

```sh
pip install -r requirements.txt
```

---

## Usage

See the docstrings in each module and the example notebooks.

### Quick example — neutral sphere

```python
from miescat import miescat

wl_0   = 0.83e-6   # wavelength in vacuum [m]
m_m    = 1.0        # refractive index of medium
d_p    = 0.4e-6    # particle diameter [m]
dens_p = 2.7        # particle density [g/cm3]

Qsca, Qext, Qabs, Qback, S11, S22, MSC, MEC, MAC = \
    miescat(wl_0, m_m, d_p, m_p_real=1.5, m_p_imag=0.0, dens_p=dens_p, nang=3)
```

### Quick example — charged sphere

```python
from miescat_charged import miescat_charged

surf_potential, Qsca, Qext, Qabs, Qback, S11, S22, MSC, MEC, MAC = \
    miescat_charged(wl_0=0.834e-6, d_p=10e-9,
                    m_p_real=1.5, m_p_imag=0.0,
                    n_e=100, gamma_fac=0.1, dens_p=2.7, nang=3)
```

### Quick example — CAS complex amplitudes

```python
from mie_complex_amplitudes import mie_complex_amplitudes

S_fwd_real, S_fwd_imag, S_bak_real, S_bak_imag = \
    mie_complex_amplitudes(wl_0=0.6328, m_m=1.33154, d_p=0.803,
                           m_p_real=1.585, m_p_imag=0.0, nang=3)
```

---

## Refactoring notes (v0.2.0)

The codebase was refactored to improve **performance** and **readability** while preserving all public function signatures and numerical results.

Key changes:
- Extracted shared computation into `_mie_core.py` — eliminates code duplication across the three modules.
- Replaced element-wise `for` loops with NumPy vectorized operations for `fn1/fn2/sg` arrays, partial wave coefficients `a/b`, and scattering amplitudes `S1/S2`.
- Replaced `np.zeros(n) + 1j*np.zeros(n)` with `np.zeros(n, dtype=complex)`.
- Moved `import numpy as np` from inside functions to module top level.
- Promoted physical constants in `miescat_charged.py` to module-level constants.
- Adopted NumPy-style docstrings throughout.

Numerical equivalence was verified with `test_refactoring.py` (101 checks, `rtol=1e-12`, all passed).

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## References

- Mie theory and computational methods
  - Bohren, C. F., & Huffman, D. R. (2008). *Absorption and scattering of light by small particles*. John Wiley & Sons.
  - Mishchenko, M. I., Travis, L. D., & Lacis, A. A. (2002). *Scattering, absorption, and emission of light by small particles*. Cambridge University Press.

- Scattering by charged spheres
  - Klacka, J., & Kocifaj, M. (2010). On the scattering of electromagnetic waves by a charged sphere. *Progress In Electromagnetics Research*, 109, 17–35.

- Complex Amplitude Sensing (CAS)
  - Moteki, N. (2021). Measuring the complex forward-scattering amplitude of single particles by self-reference interferometry: CAS-v1 protocol. *Optics Express*, 29(13), 20688–20714. https://doi.org/10.1364/OE.423175
  - Moteki, N., & Adachi, K. (2024). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol. *Optics Express*, 32(21), 36500–36522. https://doi.org/10.1364/OE.533776

---

## Author

Name: Nobuhiro Moteki
GitHub: [@NobuhiroMoteki](https://github.com/NobuhiroMoteki)
Email: nobuhiro.moteki@gmail.com
