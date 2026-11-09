# -*- coding: utf-8 -*-
"""One-point calibration of the chamber thermistor against a reference thermometer (read-only).

Reads ai2 for a few seconds (no digital-output task: the valves are never touched), then prints
the current reading and the R0 that makes it equal to the reference temperature.
Enter that R0 in the GUI field "R0 (サーミスタ)" (or set R0_TH in cloudchamber/conversions.py).
Usage:  python calib_thermistor.py <T_ref [C]> [device] [model] [seconds]
        e.g. python calib_thermistor.py 22.8 Dev2 USB-6210 10
"""
import sys

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader

from cloudchamber import conversions as cv
from cloudchamber.daq import DAQ_MODELS

RATE = 100.0


def main():
    T_ref = float(sys.argv[1]) + 273.15
    device = sys.argv[2] if len(sys.argv) > 2 else "Dev2"
    model = sys.argv[3] if len(sys.argv) > 3 else "USB-6210"
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    n = int(RATE * seconds)
    chan, vmin, vmax = DAQ_MODELS[model]["ai"][2]
    rin = DAQ_MODELS[model]["rin"]
    with nidaqmx.Task() as ai:
        ai.ai_channels.add_ai_voltage_chan(f"{device}/{chan}", terminal_config=TerminalConfiguration.RSE,
                                           min_val=vmin, max_val=vmax)
        ai.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.CONTINUOUS,
                                      samps_per_chan=max(n, 1000))
        buf = np.empty(n)
        ai.start()
        AnalogSingleChannelReader(ai.in_stream).read_many_sample(
            buf, number_of_samples_per_channel=n, timeout=seconds + 10.0)

    v = float(np.median(buf))
    T_now = float(cv.thermistor_k(v, cv.R0_TH, rin))
    r0 = float(cv.thermistor_r0(v, T_ref, rin))
    print(f"{device}/{chan} ({model}): median {v:.5f} V, sd {buf.std() * 1e3:.3f} mV over {seconds:g} s")
    print(f"  reading with R0 = {cv.R0_TH:.0f} ohm : {T_now - 273.15:7.3f} C")
    print(f"  reference               : {T_ref - 273.15:7.3f} C  (difference {T_now - T_ref:+.3f} K)")
    print(f"  -> calibrated R0 = {r0:.0f} ohm")


if __name__ == "__main__":
    main()
