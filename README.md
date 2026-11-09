# Cloud Chamber Experiment

An expansion cloud chamber that measures the particle number concentration *N* and the water vapor
mixing ratio *q<sub>w</sub>* of room air once a minute. Sample air is expanded in a glass flask. The
droplets form on the particles from the sample's own water vapor, and their laser extinction is recorded.
A physical model of the expansion, the activation and growth of the droplets and the light reaching the
photodiode is inverted by neural posterior estimation, a form of simulation-based inference. The method
needs no working fluid and no calibration against a reference particle counter or hygrometer.

The instrument is built from general-purpose parts. Undergraduate students have built instruments of
the same design in a laboratory course at the University of Tokyo. The course material is in
Japanese ([README_ja.md](README_ja.md), `Docs/KankyoJissu2026/`).

## Paper and data

- N. Moteki, *Working-fluid-free measurement of particle number concentration and water vapor from the
  laser-extinction curve of an expansion cloud by simulation-based inference*, submitted to the Journal
  of Aerosol Science. The manuscript and the supplement are in `Paper/JAerosolSci/`.
- Data: N. Moteki (2026), *Data for: Working-fluid-free measurement of particle number concentration and
  water vapor ...*, Zenodo, https://doi.org/10.5281/zenodo.24070205. The archive holds the raw signals of
  the 3132 cycles of 31 Oct to 2 Nov 2026, the per-cycle records, the CPC record and the estimates used
  in the paper.

Comparison with the reference instruments in 52 h of indoor air (3114 cycles, each analyzed by an
estimator trained without its day):

| Quantity | Reference | Median ratio (IQR) | Pearson's *r* |
|---|---|---|---|
| *q<sub>w</sub>* | Chilled-mirror hygrometer (Tokyo Opto-Electronics DPH-203B) | 0.997 (0.987–1.007) | 0.992 |
| *N* | Condensation particle counter (TSI 3022A) | 0.995 (0.950–1.046) | 0.997 |

## Repository layout

| Path | Content |
|---|---|
| `Python/cloudchamber/` | Instrument control and analysis: data acquisition (`daq.py`), valve sequence (`controller.py`), GUI (`gui.py`), forward model (`activation.py`, `mie.py`), estimator of the paper (`sbi_shape4.py`, `shape_estimator.py`), CPC interface (`cpc.py`) |
| `Python/models/` | Trained estimators (`npe_shape4_*.pt`), pressure-curve libraries and the results of the sensitivity tests of the paper. The GUI uses `npe_shape4_30000.pt`, trained on all three days |
| `Python/run_sbi_shape4.py` | Simulations and training of an estimator |
| `Python/reanalyze.py` | Analysis of recorded cycles with a given estimator |
| `Python/paper_*.py` | Figures, tables and numbers of the paper and its supplement |
| `Python/MANUAL.md` | Operation manual of the instrument (Japanese) |
| `Python/external/MieScat_Py/` | Mie scattering library (MIT License) |
| `Paper/JAerosolSci/` | Manuscript, supplement, tables and figures (LuaLaTeX) |
| `Docs/KankyoJissu2026/`, `Docs/archive/` | Course text (Japanese) and its earlier version |
| `Docs/costs/` | Price surveys of the parts (source of Table S1 of the supplement) |
| `Docs/figures/`, `Docs/reference/` | Circuit diagrams and reference photographs |
| `Python/2021_data/`, `Python/配布コード/`, `Python/Master/`, scripts directly under `Python/` | Data and code of earlier course exercises (Japanese) |

## Reproducing the results of the paper

1. Create the environment with `conda env create -f environment.yml` (Python 3.12, PyTorch, sbi 0.27.0).
   With [uv](https://docs.astral.sh/uv/), scripts can also be run as
   `uv run --no-project --python 3.12 --with numpy --with scipy --with pandas --with matplotlib --with numba --with torch --with sbi==0.27.0 python <script>`.
2. Download the zip from Zenodo and put its contents under `data/` at the top of the repository:

   ```bash
   unzip CloudChamber_JAS2026_data.zip
   mkdir -p data/cpc data/reanalysis
   mv CloudChamber_JAS2026_data/raw/raw_* data/
   mv CloudChamber_JAS2026_data/cycles/*.csv data/
   mv CloudChamber_JAS2026_data/cpc/*.csv data/cpc/
   mv CloudChamber_JAS2026_data/estimates/*.csv data/reanalysis/
   ```

3. Run the scripts from `Python/`. For example, `python paper_fig_validation.py` makes Fig. 6 and prints
   the numbers of the table above. Each `paper_*.py` names its outputs in its docstring.
4. The estimates can be recomputed from the raw files. For 31 Oct:

   ```bash
   python reanalyze.py 2026-10-31T00:00 2026-11-01T00:00 ../data/reanalysis/reanalysis_nopcorr-lodo1031.csv models/npe_shape4_nopcorr-lodo1031_30000.pt
   ```

   The estimator of each day is listed in the README of the data archive. The simulations used for
   training are not stored. Scripts that retrain an estimator simulate them again (`run_sbi_shape4.py`).

## Running the instrument

The instrument runs on Windows with an NI USB-6210 data acquisition device (NI-DAQmx) and the conda
environment above. Start the GUI with `Python\run_gui.bat`. A simulation mode runs without hardware.
Wiring, start-up checks, the automatic 60-s cycle and the output files are described in
`Python/MANUAL.md` (Japanese).

## Estimator versions

The GUI uses the newest trained version. Version 4 (`sbi_shape4.py`) is the one of the paper. It uses the
extinction efficiency corrected for the light scattered into the photodiode and the recorded pressure
curves, without a correction of the pressure-sensor response. Earlier versions (`sbi_shape.py` to
`sbi_shape3.py`) are kept for reference.

## License

MIT License (`LICENSE`). `Python/external/MieScat_Py/` has its own MIT License.

## Contact

Nobuhiro Moteki, Tokyo Metropolitan University (moteki@tmu.ac.jp)
