# -*- coding: utf-8 -*-
"""Real-time logger for the TSI 3022A CPC over RS-232 (read-only commands; see cloudchamber/cpc.py).

Polls the front-panel concentration (RD), the 1-s count buffer (RB) and the flow (R4) every
--interval seconds, and the instrument status (READY, liquid level, temperatures) every
--status-every polls; writes one CSV per day, data/cpc/cpc_YYYYMMDD.csv, flushed on every line.
Reconnects after a serial error.

The GUI records the CPC itself when a CPC port is set (same files, plus the 1-min mean in the
cloud_data CSV); use this script only when the GUI is not running, since one port has one owner.

Usage:
  python cpc_logger.py --port COM3              # log until Ctrl+C
  python cpc_logger.py --port COM3 --test       # print every read command once and exit
"""
import argparse
import os

from cloudchamber.cpc import CPC3022A, CPCRecorder, READ_COMMANDS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, os.pardir, "data", "cpc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--parity", choices=("E", "N"), default="E")
    ap.add_argument("--interval", type=float, default=1.0, help="[s] polling period")
    ap.add_argument("--status-every", type=int, default=60)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        cpc = CPC3022A(args.port, baudrate=args.baud, parity=args.parity)
        for c, name in READ_COMMANDS.items():
            try:
                print(f"{c:3s} {name:16s} {cpc.query(c)}")
            except Exception as e:
                print(f"{c:3s} {name:16s} -- {e}")
        cpc.close()
        return

    rec = CPCRecorder(args.port, args.out, interval=args.interval, status_every=args.status_every,
                      keep=60, baudrate=args.baud, parity=args.parity,
                      log=lambda s: print(s, flush=True))
    rec.start()
    try:
        while rec.is_alive():
            rec.join(0.5)
    except KeyboardInterrupt:
        rec.stop()
        rec.join(5)


if __name__ == "__main__":
    main()
