# -*- coding: utf-8 -*-
"""Read-only sensor check: reads ai0-ai4 for a few seconds and prints voltages and converted values.

No digital-output task is created, so the valves are never touched.
Usage:  python check_daq.py [device] [model] [seconds]
        e.g. python check_daq.py Dev2 USB-6210 3   (these are the defaults)
"""
import sys

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader

from cloudchamber import conversions as cv
from cloudchamber.daq import DAQ_MODELS

RATE = 100.0


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "Dev2"
    model = sys.argv[2] if len(sys.argv) > 2 else "USB-6210"
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    n = int(RATE * seconds)
    with nidaqmx.Task() as ai:
        for chan, vmin, vmax in DAQ_MODELS[model]["ai"]:
            ai.ai_channels.add_ai_voltage_chan(f"{device}/{chan}",
                                               terminal_config=TerminalConfiguration.RSE,
                                               min_val=vmin, max_val=vmax)
        ai.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.CONTINUOUS,
                                      samps_per_chan=max(n, 1000))
        buf = np.empty((5, n))
        ai.start()
        AnalogMultiChannelReader(ai.in_stream).read_many_sample(
            buf, number_of_samples_per_channel=n, timeout=seconds + 10.0)

    print(f"{device} ({model}), {n} samples/ch at {RATE:g} Hz")
    for (chan, vmin, vmax), name, v in zip(DAQ_MODELS[model]["ai"],
                                           ("photodiode", "pressure", "thermistor", "RH out",
                                            "dew point"), buf):
        print(f"  {chan} {name:10s} range +-{vmax:g} V: mean {v.mean():.5f} V  sd {v.std():.5f} V  "
              f"min {v.min():.5f}  max {v.max():.5f}  distinct values {np.unique(v).size}")
    p = cv.pressure_pa(buf[1]) / 100
    T = cv.thermistor_k(buf[2], rin=DAQ_MODELS[model]["rin"]) - 273.15
    print(f"  pressure    median {np.median(p):8.2f} hPa  sd {p.std():.3f} hPa")
    print(f"  temperature median {np.median(T):8.3f} C    sd {T.std():.4f} C")
    rh = cv.hygrometer_rh(buf[3]) * 100
    print(f"  hygrometer  median {np.median(rh):8.2f} %RH  sd {rh.std():.3f} %RH")
    td = cv.dewpoint_c(buf[4])
    qw = cv.qw_from_dewpoint(np.median(td), np.median(cv.pressure_pa(buf[1])))
    print(f"  dew point   median {np.median(td):8.3f} C    sd {td.std():.4f} C   -> qw {qw*1e3:.3f} g/kg"
          f"  (Td = {cv.TD_SLOPE} C/mV * V + {cv.TD_OFFSET} C)")
    print("  (channels all at the same voltage usually means the inputs are not connected)")


if __name__ == "__main__":
    main()
