# -*- coding: utf-8 -*-
"""DAQ back-ends: NI USB DAQ via nidaqmx, and a simulated chamber for testing without hardware.

Signals (same channel numbers on every model, single-ended/RSE):
  ai0 = photodiode (up to ~8 V), ai1 = Setra 276 pressure (0.1-5.1 V),
  ai2 = thermistor divider (~1-2.5 V), ai3 = chilled-mirror hygrometer RH (0-1 V = 0-100 %),
  ai4 = hygrometer dew-point voltage output (~10 mV/C, about +-1 V)
  two digital output lines: first = V1 (inlet), second = V2 (to vacuum pump); True = open
"""
import time

import numpy as np

from . import conversions as cv

SAFE_VALVES = (True, False)  # V1 open, V2 closed: chamber vented, pump isolated

# per model: AI (channel, min V, max V), the two valve DO lines and the AI input impedance [ohm]
DAQ_MODELS = {
    # 11-bit single-ended, fixed +-10 V range; DO on port0 (5 V)
    "USB-6008/6009": dict(ai=[("ai0", -10.0, 10.0), ("ai1", -10.0, 10.0), ("ai2", -10.0, 10.0),
                              ("ai3", -10.0, 10.0), ("ai4", -10.0, 10.0)],
                          do="port0/line0:1", rin=cv.RIN_ADC),
    # 16-bit, selectable ranges; the 4 static DO lines are port1 (PFI4-7); Rin > 10 GOhm
    "USB-6210": dict(ai=[("ai0", -10.0, 10.0), ("ai1", -5.0, 5.0), ("ai2", -5.0, 5.0),
                         ("ai3", -1.0, 1.0), ("ai4", -1.0, 1.0)],
                     do="port1/line0:1", rin=np.inf),
}


class NiDaq:
    """Hardware-timed continuous AI at `rate` [Hz] and software-timed valve DO."""

    def __init__(self, device="Dev2", rate=100.0, model="USB-6210"):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, LineGrouping, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        cfg = DAQ_MODELS[model]
        self.rate = rate
        self.rin = cfg["rin"]
        self._ai = nidaqmx.Task()
        self._do = nidaqmx.Task()
        try:
            for chan, vmin, vmax in cfg["ai"]:
                self._ai.ai_channels.add_ai_voltage_chan(
                    f"{device}/{chan}", terminal_config=TerminalConfiguration.RSE,
                    min_val=vmin, max_val=vmax)
            self._ai.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=int(rate * 10))
            self._do.do_channels.add_do_chan(f"{device}/{cfg['do']}",
                                             line_grouping=LineGrouping.CHAN_PER_LINE)
        except Exception:
            self.close()
            raise
        self._reader = AnalogMultiChannelReader(self._ai.in_stream)

    def start(self):
        self.set_valves(*SAFE_VALVES)
        self._ai.start()

    def read(self, n):
        """Block until n new samples per channel are available; returns array (5, n) [V]."""
        buf = np.empty((5, n))
        self._reader.read_many_sample(buf, number_of_samples_per_channel=n, timeout=10.0)
        return buf

    def set_valves(self, v1, v2):
        self._do.write([bool(v1), bool(v2)], auto_start=True)

    def close(self):
        try:
            self.set_valves(*SAFE_VALVES)
        except Exception:
            pass
        for task in (self._ai, self._do):
            try:
                task.close()
            except Exception:
                pass


class SimulatedDaq:
    """Synthetic sensor voltages from the forward model; qw and nd drift slowly.

    speed > 1 runs faster than real time (for tests).
    """

    def __init__(self, rate=100.0, speed=1.0, seed=0, T_amb=299.0, p_amb=1002e2,
                 p_offset=441e2, tau_vac=3.95, valve_lag=0.2, pd_volt=2.0):
        self.rate, self.speed = rate, speed
        self.rng = np.random.default_rng(seed)
        self.T_amb, self.p_amb = T_amb, p_amb
        self.p_offset, self.tau_vac, self.valve_lag = p_offset, tau_vac, valve_lag
        self.pd_volt = pd_volt
        self.qw, self.log10nd = 0.012, 10.0
        self.truth = []  # (sample index of V2 opening, qw, log10nd)
        self._v1, self._v2 = SAFE_VALVES
        self._i = 0
        self._p = p_amb
        self._t_open = None
        self._sim = None
        self._t_next = None

    def start(self):
        self._t_next = time.perf_counter()

    def set_valves(self, v1, v2):
        if v2 and not self._v2 and not v1:
            self._open_v2()
        if not v2:
            self._sim = None
        self._v1, self._v2 = bool(v1), bool(v2)

    def _open_v2(self):
        from .forward import simulate
        self.qw = float(np.clip(self.qw + self.rng.normal(0, 0.0005), 0.007, 0.016))
        self.log10nd = float(np.clip(self.log10nd + self.rng.normal(0, 0.1), 9.0, 11.0))
        self.truth.append((self._i, self.qw, self.log10nd))
        self._sim = simulate(self.qw, 10 ** self.log10nd, T_ini=self.T_amb, p_ini=self._p,
                             p_offset=self.p_offset, tau_vac=self.tau_vac, t_end=10.0,
                             out_dt=1.0 / self.rate)
        self._t_open = self._i / self.rate

    def read(self, n):
        self._t_next += n / self.rate / self.speed
        delay = self._t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        from .constants import A_ES, B_ES, EPS
        out = np.empty((5, n))
        rh_amb = self.qw * self.p_amb / (EPS * A_ES * np.exp(-B_ES / self.T_amb))
        e = self.qw * self.p_amb / (EPS + self.qw)  # invert the Magnus formula for the dew point
        g = np.log(e / 611.2)
        td_amb = 243.12 * g / (17.62 - g)
        for k in range(n):
            t = self._i / self.rate
            tr = 1.0
            if self._sim is not None:
                j = int(round((t - self._t_open - self.valve_lag) * self.rate))
                if j >= 0:
                    j = min(j, self._sim["p"].size - 1)
                    self._p, tr = self._sim["p"][j], self._sim["Tr"][j]
            elif self._v1:  # vented: relax to ambient
                self._p += (self.p_amb - self._p) / (0.5 * self.rate)
            out[0, k] = self.pd_volt * tr + self.rng.normal(0, 1e-3)
            out[1, k] = cv.pressure_to_volt(self._p) + self.rng.normal(0, 1e-4)
            out[2, k] = cv.thermistor_to_volt(self.T_amb) + self.rng.normal(0, 2e-4)
            out[3, k] = cv.rh_to_volt(rh_amb) + self.rng.normal(0, 2e-4)
            out[4, k] = cv.dewpoint_to_volt(td_amb) + self.rng.normal(0, 2e-4)
            self._i += 1
        return out

    def close(self):
        pass
