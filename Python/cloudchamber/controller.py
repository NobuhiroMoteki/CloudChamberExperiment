# -*- coding: utf-8 -*-
"""Measurement and valve-control loop (Python port of CloudChamberOperation_AutoValve.vi).

Valve cycle (Docs Sec. 4.5), each phase `durations[i]` seconds:
  1 air exchange (V1 open, V2 open), 2 pressure relaxation (V1 open), 3 sealed (both closed),
  4 decompression (V2 open), 5 vent (V1 open), 6 wait (V1 open).
At the end of the sealed phase: I0 = median PD voltage of the last 10 s (NaN if less data),
T_ini and p_ini = medians of the last 1 s. During decompression, p(t) and Tr(t) = V_PD/I0 are
recorded; afterwards they are analysed in a worker thread and passed to the estimator.
With a CPCRecorder, each cycle row also gets the CPC median (counts/flow) over one full valve cycle
ending with the decompression (cpc_cm3, NaN without readings), and the number of 1-s readings in it
(cpc_n). Cycles with I0 < I0_MIN are not estimated ("no_laser").
"""
import os
import queue
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import conversions as cv
from .constants import A_ES, B_ES, EPS
from .analysis import analyze_decompression
from .daq import SAFE_VALVES

PHASES = [  # (name, V1, V2)
    ("空気置換", True, True),
    ("気圧緩和", True, False),
    ("密閉", False, False),
    ("減圧", False, True),
    ("開放", True, False),
    ("待機", True, False),
]
I_SEALED, I_DECOMP = 2, 3

CSV_COLUMNS = ["Time_stamp", "datetime", "T_ini", "p_ini", "p_offset", "tau_vac", "p_c",
               "zeta_moist", "zeta_turbid", "t0", "I0", "qw_gkg", "qw_lo", "qw_hi",
               "log10nd", "log10nd_lo", "log10nd_hi", "ref_RH", "qw_ref_gkg", "ref_Td",
               "qw_ref_td_gkg", "nd_cm3", "nd_cm3_lo", "nd_cm3_hi", "Td_est", "Td_est_lo",
               "Td_est_hi", "t_s", "t0_ramp2", "p_offset2", "tau_vac1", "frac1", "tau_vac2",
               "t_open", "p_rms", "t_s_shape", "cpc_cm3", "cpc_n", "estimator", "flags"]
# The CPC median is taken over one full valve cycle, which assumes the CPC samples room air through
# its own inlet (from 2026-10-31). While it shared the chamber's sample line, the V1 openings
# depressed its reading by up to ~90 % for several seconds, and only the V1-closed part (from 3 s
# before the decompression to its end) was used.
# Below this I0 [V] the laser is off or blocked (dark level ~0.05 V, normal ~7 V): the cycle is
# saved with the flag "no_laser" and not passed to the estimator (noise-only Tr is far outside the
# training set, and the posterior sampling then hardly terminates).
I0_MIN = 1.0
# ramp + double-exponential pressure fit and Tr-shape onset (filled by ShapeSBIEstimator)
RAMP2_COLUMNS = {"t0_ramp2": 1.0, "p_offset2": 1e-2, "tau_vac1": 1.0, "frac1": 1.0, "tau_vac2": 1.0,
                 "t_open": 1.0, "p_rms": 1e-2, "t_s_shape": 1.0}  # factor to CSV units (Pa -> hPa)


class Controller(threading.Thread):
    """Runs the valve cycle forever until stop() is called.

    With manual=True no cycle is run: the valves follow request_valves() and only live data are
    produced (I0 is then the running median of the last <= 10 s, so I/I0 can be monitored).

    Live data and results are put on `self.out` as tuples:
      ("live", dict(t, pd, p, T, rh, td, tr)), ("phase", (index, name, v1, v2)), ("cycle", dict),
      ("error", str)
    """

    def __init__(self, daq, estimator=None, durations=(10,) * 6, tr_threshold=0.985,
                 r0=cv.R0_TH, data_dir="data", chunk=10, manual=False, cpc=None):
        super().__init__(daemon=True)
        self.daq, self.estimator, self.manual = daq, estimator, manual
        self.cpc = cpc  # CPCRecorder or None
        self._valve_req = None
        self.rate = daq.rate
        self.n_phase = [int(round(d * self.rate)) for d in durations]
        self.tr_threshold, self.r0, self.chunk = tr_threshold, r0, chunk
        self.rin = getattr(daq, "rin", cv.RIN_ADC)
        self.data_dir = data_dir
        self.out = queue.Queue()
        self._stop_event = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1)

        n10, n1 = int(10 * self.rate), int(1 * self.rate)
        self._pd10 = deque(maxlen=n10)
        self._rh10 = deque(maxlen=n10)
        self._td10 = deque(maxlen=n10)
        self._p1 = deque(maxlen=n1)
        self._T1 = deque(maxlen=n1)
        self.I0 = np.nan
        self._latched = {}
        self._rec = None

    def stop(self):
        self._stop_event.set()

    def request_valves(self, v1, v2):
        """Manual mode only: valve state to apply before the next data chunk."""
        self._valve_req = (bool(v1), bool(v2))

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self.daq.start()
            t_start = time.time()
            if self.manual:
                self._loop_manual(t_start)
            else:
                self._loop_auto(t_start)
        except Exception:  # report to the GUI instead of dying silently
            self.out.put(("error", traceback.format_exc()))
        finally:
            self.daq.close()
            self._pool.shutdown(wait=True)

    def _read_chunk(self, t_start, i_sample):
        v = self.daq.read(self.chunk)
        pd = v[0]
        p = cv.pressure_pa(v[1])
        T = cv.thermistor_k(v[2], self.r0, self.rin)
        rh = cv.hygrometer_rh(v[3])
        td = cv.dewpoint_c(v[4])
        t = t_start + (i_sample + np.arange(self.chunk)) / self.rate
        return t, pd, p, T, rh, td

    def _loop_manual(self, t_start):
        state = SAFE_VALVES  # applied by daq.start()
        self.out.put(("phase", (-1, "手動運転", *state)))
        i_sample = 0
        while not self._stop_event.is_set():
            req, self._valve_req = self._valve_req, None
            if req is not None and req != state:
                self.daq.set_valves(*req)
                state = req
                self.out.put(("phase", (-1, "手動運転", *state)))
            t, pd, p, T, rh, td = self._read_chunk(t_start, i_sample)
            self._pd10.extend(pd)
            if len(self._pd10) >= self.rate:  # monitoring only: >= 1 s of data is enough
                self.I0 = float(np.median(self._pd10))
            i_sample += self.chunk
            self.out.put(("live", dict(t=t, pd=pd, p=p, T=T, rh=rh, td=td, tr=pd / self.I0)))

    def _loop_auto(self, t_start):
        i_sample, phase, k_phase = 0, -1, 0
        while not self._stop_event.is_set():
            t, pd, p, T, rh, td = self._read_chunk(t_start, i_sample)
            tr = pd / self.I0
            for k in range(self.chunk):
                if phase < 0 or k_phase >= self.n_phase[phase]:
                    if phase == I_SEALED:
                        self._latch()
                    elif phase == I_DECOMP:
                        self._finish_decompression()
                    phase, k_phase = (phase + 1) % len(PHASES), 0
                    name, v1, v2 = PHASES[phase]
                    self.daq.set_valves(v1, v2)
                    self.out.put(("phase", (phase, name, v1, v2)))
                    if phase == I_DECOMP:
                        self._rec = dict(t=[], p=[], tr=[], T=[], t_wall=t[k])
                self._pd10.append(pd[k])
                self._rh10.append(rh[k])
                self._td10.append(td[k])
                self._p1.append(p[k])
                self._T1.append(T[k])
                if phase == I_DECOMP:
                    tr_k = pd[k] / self.I0
                    tr[k] = tr_k
                    self._rec["t"].append(k_phase / self.rate)
                    self._rec["p"].append(p[k])
                    self._rec["tr"].append(tr_k)
                    self._rec["T"].append(T[k])
                k_phase += 1
            i_sample += self.chunk
            self.out.put(("live", dict(t=t, pd=pd, p=p, T=T, rh=rh, td=td, tr=tr)))

    # ------------------------------------------------------------------ phase events
    def _latch(self):
        full = len(self._pd10) == self._pd10.maxlen
        self.I0 = float(np.median(self._pd10)) if full else np.nan
        T_ini, p_ini = float(np.median(self._T1)), float(np.median(self._p1))
        # reference hygrometer, medians over the sealed phase. The dew point gives qw directly;
        # the RH output is converted assuming it is relative to T_ini (kept for comparison).
        ref_rh = float(np.median(self._rh10)) if self._rh10 else np.nan
        ref_td = float(np.median(self._td10)) if self._td10 else np.nan
        qw_ref = EPS * ref_rh * A_ES * np.exp(-B_ES / T_ini) / p_ini
        self._latched = dict(I0=self.I0, T_ini=T_ini, p_ini=p_ini, ref_RH=ref_rh,
                             qw_ref_gkg=qw_ref * 1e3, ref_Td=ref_td,
                             qw_ref_td_gkg=float(cv.qw_from_dewpoint(ref_td, p_ini)) * 1e3)

    def _finish_decompression(self):
        rec = {key: np.asarray(val) if isinstance(val, list) else val
               for key, val in self._rec.items()}
        rec.update(self._latched)
        # CPC taken now, not in the worker, so a backlog there cannot outlast the CPC buffer
        t_end = rec["t_wall"] + self.n_phase[I_DECOMP] / self.rate
        t_cycle = sum(self.n_phase) / self.rate
        rec["cpc_cm3"], rec["cpc_n"] = (self.cpc.median(t_end - t_cycle, t_end)
                                        if self.cpc is not None else (np.nan, 0))
        self._pool.submit(self._analyse, rec)

    def _analyse(self, rec):
        # A failure here must not stop a long unattended run: the cycle is still saved (with NaNs
        # and a flag) and the traceback goes to the GUI log and to data_dir/warnings_YYYYMMDD.log.
        try:
            res = analyze_decompression(rec["t"], rec["p"], rec["tr"], rec["p_ini"],
                                        tr_threshold=self.tr_threshold)
        except Exception:
            self._warn(rec, "analysis")
            res = dict(t0=np.nan, p_offset=np.nan, tau_vac=np.nan, t_c=np.nan, t_s=np.nan,
                       p_c=np.nan, zeta_moist=np.nan, zeta_turbid=np.nan, flags=["analysis_error"])
        cycle = dict(rec, **res)
        if not cycle["I0"] >= I0_MIN:
            cycle["flags"] = cycle["flags"] + ["no_laser"]
        elif self.estimator is not None:
            try:
                cycle.update(self.estimator(cycle))
                reason = cycle.pop("sbi_reason", None)
                if reason:
                    cycle["flags"] = cycle["flags"] + [reason]
            except Exception:
                self._warn(rec, "estimator")
                cycle["flags"] = cycle["flags"] + ["estimator_error"]
        for s in ("", "_lo", "_hi"):  # aerosol number per cm3 at the initial (sealed) state
            cycle[f"nd_cm3{s}"] = float(cv.number_per_cm3(10 ** cycle.get(f"log10nd{s}", np.nan),
                                                          cycle["p_ini"], cycle["T_ini"]))
        for s, q in (("", "qw_gkg"), ("_lo", "qw_lo"), ("_hi", "qw_hi")):  # dew point of the estimate
            cycle[f"Td_est{s}"] = float(cv.dewpoint_from_qw(cycle.get(q, np.nan) * 1e-3,
                                                             cycle["p_ini"]))
        try:
            self._save(cycle)
        except Exception:
            self._warn(rec, "save")
        self.out.put(("cycle", cycle))

    def _warn(self, rec, where):
        msg = f"{datetime.now():%Y-%m-%d %H:%M:%S} {where}: {traceback.format_exc()}"
        self.out.put(("warning", msg))
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(os.path.join(self.data_dir, f"warnings_{datetime.now():%Y%m%d}.log"), "a",
                      encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ output files
    def _save(self, c):
        stamp = datetime.fromtimestamp(c["t_wall"], tz=timezone.utc).astimezone()  # local, tz-aware
        os.makedirs(self.data_dir, exist_ok=True)
        header = ",".join(CSV_COLUMNS)
        path = os.path.join(self.data_dir, f"cloud_data_{stamp:%Y%m%d}.csv")
        k = 1
        while os.path.exists(path):  # never append rows to a file written with other columns
            with open(path, encoding="utf-8") as f:
                if f.readline().rstrip("\n") == header:
                    break
            k += 1
            path = os.path.join(self.data_dir, f"cloud_data_{stamp:%Y%m%d}_v{k}.csv")
        new = not os.path.exists(path)
        hpa = lambda x: x * 1e-2  # noqa: E731
        row = [f"{c['t_wall']:.3f}", stamp.isoformat(timespec="seconds"),
               f"{c['T_ini']:.4f}", f"{hpa(c['p_ini']):.4f}", f"{hpa(c['p_offset']):.4f}",
               f"{c['tau_vac']:.4f}", f"{hpa(c['p_c']):.4f}", f"{c['zeta_moist']:.5f}",
               f"{c['zeta_turbid']:.5f}", f"{c['t0']:.4f}", f"{c['I0']:.5f}"]
        row += [f"{c.get(k, np.nan):.5g}" for k in CSV_COLUMNS[11:17]]
        row += [f"{c['ref_RH']:.5f}", f"{c['qw_ref_gkg']:.4f}", f"{c['ref_Td']:.3f}",
                f"{c['qw_ref_td_gkg']:.4f}"]
        row += [f"{c.get(k, np.nan):.1f}" for k in ("nd_cm3", "nd_cm3_lo", "nd_cm3_hi")]
        row += [f"{c.get(k, np.nan):.3f}" for k in ("Td_est", "Td_est_lo", "Td_est_hi")]
        row.append(f"{c.get('t_s', np.nan):.4f}")
        row += [f"{c.get(k, np.nan) * s:.5g}" for k, s in RAMP2_COLUMNS.items()]
        row += [f"{c.get('cpc_cm3', np.nan):.1f}", f"{c.get('cpc_n', 0)}"]
        row.append(getattr(self.estimator, "name", type(self.estimator).__name__)
                   if self.estimator is not None else "")
        row.append("|".join(c["flags"]))
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(header + "\n")
            f.write(",".join(row) + "\n")
        # raw decompression series, for re-analysis and SBI validation (pressures in hPa, as in the CSV).
        # Series are float32 (ample for the 12-bit ADC); t_wall stays float64 to keep sub-second UNIX time.
        raw_dir = os.path.join(self.data_dir, f"raw_{stamp:%Y%m%d}")
        os.makedirs(raw_dir, exist_ok=True)
        f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
        np.savez_compressed(os.path.join(raw_dir, f"decomp_{stamp:%Y%m%d_%H%M%S}.npz"),
                            t=f32(c["t"]), p=f32(hpa(c["p"])), tr=f32(c["tr"]), T=f32(c["T"]),
                            t_wall=c["t_wall"], I0=c["I0"], T_ini=c["T_ini"], p_ini=hpa(c["p_ini"]),
                            ref_RH=c["ref_RH"], ref_Td=c["ref_Td"], r0=self.r0, rin=self.rin)
