# -*- coding: utf-8 -*-
"""TSI 3022A CPC: real-time RS-232 reading, Aerosol Instrument Manager CSV exports, logger CSVs and
comparison with the cloud-chamber cycles.

The CPC shares the sample line with the chamber, so its readings carry a 60 s periodic artefact
from the line-pressure changes at each valve switch. smooth() removes it with a 60 s boxcar (exactly
one valve cycle) followed by a centred running mean.

RS-232 (manual Ch. 8): 9-pin COM PORT, default 9600 baud, even parity, 7 data bits, 1 stop bit
(TSICPC.BAS opens "COM1:...,E,7,1"); ASCII commands terminated by CR LF, one-line replies, "ERROR"
for unknown commands. Only the read commands (R0-RE) are used here; the X (laser, pump, drain/fill,
flow mode), C, L and V commands change the instrument state and are deliberately not wrapped.
"""
import datetime as dt
import os
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

VALVE_PERIOD = "60s"
READ_COMMANDS = {
    "R0": "liquid_level",      # FULL / NOTFULL
    "R1": "T_condenser_C",
    "R2": "T_saturator_C",
    "R3": "T_optics_C",
    "R4": "flow_cm3s",
    "R5": "ready",             # READY / NOTREADY (temperatures, flow and liquid level in range)
    "R6": "ref_detector_V",
    "R7": "detector_V",
    "R8": "counts_200s",
    "R9": "counts_20s",
    "RA": "counts_2s",
    "RB": "counts_1s",
    "RC": "photometric_cm3",
    "RD": "conc_cm3",          # as on the front panel (photometric, single-count or live-time)
    "RE": "pump_ctrl",
}


class CPC3022A:
    """Read-only RS-232 interface. port: e.g. "COM3" (USB-serial adapter); or pass an open
    serial-like object (write(bytes), readline() -> bytes) as ser."""

    def __init__(self, port=None, baudrate=9600, parity="E", timeout=1.0, ser=None):
        if ser is None:
            import serial
            ser = serial.Serial(port, baudrate=baudrate, bytesize=serial.SEVENBITS,
                                parity=serial.PARITY_EVEN if parity == "E" else serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, timeout=timeout)
        self.ser = ser

    def close(self):
        self.ser.close()

    def query(self, cmd):
        if cmd not in READ_COMMANDS:
            raise ValueError(f"{cmd!r} is not a read command")
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()
        self.ser.write((cmd + "\r\n").encode("ascii"))
        reply = self.ser.readline().decode("ascii", errors="replace").strip()
        if not reply:
            raise TimeoutError(f"no reply to {cmd}")
        if reply == "ERROR":
            raise RuntimeError(f"CPC replied ERROR to {cmd}")
        return reply

    def read(self, cmd):
        """Reply converted to float, or the string for R0/R5."""
        reply = self.query(cmd)
        return reply if cmd in ("R0", "R5") else float(reply)

    def concentration(self):
        """Front-panel concentration [#/cm3]."""
        return self.read("RD")

    def status(self, cmds=("R0", "R1", "R2", "R3", "R4", "R5")):
        return {READ_COMMANDS[c]: self.read(c) for c in cmds}


STATUS = ("R5", "R0", "R1", "R2", "R3")
LOG_COLUMNS = (["Time_stamp", "datetime", "conc_cm3", "counts_1s", "flow_cm3s"]
               + [READ_COMMANDS[c] for c in STATUS])


def count_conc(counts, flow):
    """Concentration [#/cm3] from the 1-s counts and the sample flow [cm3/s] (NaN for an unparsable
    or non-positive flow). Unlike the front-panel value (RD) it does not drop by ~20 % when the CPC
    switches to photometric mode near 1e4 /cm3; in single-count mode RD = counts/flow within ~1 %."""
    counts = pd.to_numeric(counts, errors="coerce")
    flow = pd.to_numeric(flow, errors="coerce")
    return np.where(flow > 0, counts / np.where(flow > 0, flow, 1.0), np.nan)


class CPCRecorder(threading.Thread):
    """Polls the CPC every `interval` s (RD, RB, R4; status every `status_every` polls), appends to
    out_dir/cpc_YYYYMMDD.csv (flushed per line) and keeps the last `keep` s of concentrations in
    memory for median(): counts/flow (count_conc), or RD when the flow reply is unparsable.
    Retries opening the port every 10 s and reconnects after a serial error; `error` holds the
    last problem (None while reading normally)."""

    def __init__(self, port, out_dir, interval=1.0, status_every=60, keep=3600, baudrate=9600,
                 parity="E", log=print):
        super().__init__(daemon=True)
        self.port, self.out_dir, self.interval, self.status_every = port, out_dir, interval, status_every
        self.baudrate, self.parity, self.log = baudrate, parity, log
        self._buf = deque(maxlen=int(keep / interval) + 10)  # (UNIX time, conc)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.error = None

    def stop(self):
        self._stop_event.set()

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else (np.nan, np.nan)

    def median(self, t0, t1):
        """Median concentration [#/cm3] (robust to outliers) and number of readings with
        t0 <= time < t1 (UNIX s)."""
        with self._lock:
            c = [x for t, x in self._buf if t0 <= t < t1]
        return (float(np.median(c)) if c else np.nan), len(c)

    def _open(self):
        while not self._stop_event.is_set():
            try:
                cpc = CPC3022A(self.port, baudrate=self.baudrate, parity=self.parity)
                self.error = None
                return cpc
            except Exception as e:  # port busy / adapter unplugged: retry
                self.error = f"cannot open {self.port}: {e}"
                self.log(f"{dt.datetime.now():%H:%M:%S} {self.error}; retry in 10 s")
                self._stop_event.wait(10)
        return None

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        cpc, f, day, n = self._open(), None, None, 0
        t_next = time.time()
        try:
            while cpc is not None and not self._stop_event.is_set():
                t_next += self.interval
                now = dt.datetime.now().astimezone()
                if now.date() != day:
                    if f:
                        f.close()
                    day = now.date()
                    path = os.path.join(self.out_dir, f"cpc_{day:%Y%m%d}.csv")
                    new = not os.path.exists(path)
                    f = open(path, "a", encoding="utf-8", newline="")
                    if new:
                        f.write(",".join(LOG_COLUMNS) + "\n")
                try:
                    row = [f"{now.timestamp():.3f}", now.isoformat(timespec="seconds"),
                           cpc.query("RD"), cpc.query("RB"), cpc.query("R4")]
                    row += [cpc.query(c) if n % self.status_every == 0 else "" for c in STATUS]
                    f.write(",".join(row) + "\n")
                    f.flush()
                    conc = float(count_conc(row[3], row[4]))
                    with self._lock:
                        self._buf.append((now.timestamp(), conc if np.isfinite(conc) else float(row[2])))
                    n += 1
                except Exception as e:
                    self.error = f"read error: {e}"
                    self.log(f"{now:%H:%M:%S} {self.error}; reconnecting")
                    try:
                        cpc.close()
                    except Exception:
                        pass
                    cpc = self._open()
                    t_next = time.time()
                self._stop_event.wait(max(0.0, t_next - time.time()))
        finally:
            if f:
                f.close()
            if cpc is not None:
                cpc.close()


def read_logger_csv(paths, from_counts=True):
    """Concentration Series from CPCRecorder / cpc_logger.py output file(s) (same form as
    read_aim_csv): counts/flow (RD where the flow is unparsable), or RD with from_counts=False."""
    if isinstance(paths, str):
        paths = [paths]
    d = pd.concat([pd.read_csv(p) for p in paths])
    t = pd.to_datetime(d.Time_stamp, unit="s", utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    c = d.conc_cm3.to_numpy(float)
    if from_counts:
        cc = count_conc(d.counts_1s, d.flow_cm3s)
        c = np.where(np.isfinite(cc), cc, c)
    return pd.Series(c, index=pd.DatetimeIndex(t), name="cpc_cm3").sort_index()


def read_aim_csv(path, start_date=None):
    """Concentration [#/cm3] as a Series indexed by local time from an AIM export
    ("Start Date" header line, then "Time,Concentration" rows; midnight roll-overs handled)."""
    lines = open(path, encoding="cp932", errors="replace").read().splitlines()
    if start_date is None:
        s = next(x for x in lines if x.startswith("Start Date"))
        start_date = pd.to_datetime(s.split(",")[1], format="%m/%d/%y")
    day = pd.Timestamp(start_date)
    i0 = next(i for i, s in enumerate(lines) if s.startswith("Time,"))
    times, conc, prev = [], [], None
    for s in lines[i0 + 1:]:
        parts = s.split(",")
        if len(parts) < 2 or ":" not in parts[0]:
            break
        td = pd.Timedelta(parts[0])
        if prev is not None and td < prev:
            day += pd.Timedelta(days=1)
        prev = td
        times.append(day + td)
        conc.append(float(parts[1]))
    return pd.Series(conc, index=pd.DatetimeIndex(times), name="cpc_cm3")


def smooth(c, window="600s"):
    """Remove the valve-cycle artefact (60 s boxcar) and smooth with a centred running mean."""
    return c.rolling(VALVE_PERIOD).mean().rolling(window, center=True).mean()


def median_before(c, times, window=VALVE_PERIOD):
    """Median of the raw CPC readings in [time - window, time) for each time (NaN without readings).
    Note: while the CPC shared the chamber's sample line (until 2026-10-31), a window that includes
    the V1 openings is biased low (see the CPC note in controller.py)."""
    w = pd.Timedelta(window)
    t = c.index.values
    v = c.values
    out = []
    for x in pd.DatetimeIndex(times).values:
        i0, i1 = np.searchsorted(t, x - w), np.searchsorted(t, x)
        out.append(np.median(v[i0:i1]) if i1 > i0 else np.nan)
    return np.array(out)


def at_times(c_smooth, times):
    """Smoothed CPC concentration interpolated to the given times (NaN outside the record)."""
    t = pd.DatetimeIndex(times).asi8
    ok = c_smooth.notna()
    x = c_smooth.index[ok].asi8
    y = c_smooth.values[ok]
    return np.where((t >= x[0]) & (t <= x[-1]), np.interp(t, x, y), np.nan)
