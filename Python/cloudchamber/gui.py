# -*- coding: utf-8 -*-
"""Desktop GUI for the cloud chamber (replaces the LabVIEW front panel).

Run:  python -m cloudchamber.gui      (from the Python directory, inside the cloudchamber env)
"""
import os
import queue
import sys
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import conversions as cv
from .controller import PHASES, Controller
from .daq import DAQ_MODELS
from .features import pressure_ramp2
from .sbi_shape import SHAPE_OFFSETS

LIVE_SECONDS = 60
HISTORY_CYCLES = 24 * 60  # one day of 60 s cycles

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                os.pardir, "data")


def _fmt(x, f="{:.3f}"):
    return "—" if x is None or not np.isfinite(x) else f.format(x)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cloud Chamber — 計測制御と推定")
        self.ctrl = None
        self.daq = None
        self.cpc = None
        self.cycles = []
        self.n_warn = 0
        n = LIVE_SECONDS * 100
        self.live = {k: np.full(n, np.nan) for k in ("t", "tr", "p")}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_panel(), 0)
        layout.addWidget(self._build_plots(), 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(50)

    # ------------------------------------------------------------------ widgets
    @staticmethod
    def _scroll_tab(*widgets):
        """Tab page with the widgets stacked vertically, scrollable when the window is short."""
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setContentsMargins(4, 4, 4, 4)
        for w in widgets:
            v.addWidget(w)
        v.addStretch(1)
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(inner)
        return area

    def _build_panel(self):
        # Start/stop and the manual valve buttons stay visible; the readouts and the settings are
        # on two tabs so that neither is squeezed.
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(340)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        box = QtWidgets.QGroupBox("開始前に設定するパラメータ")
        form = QtWidgets.QFormLayout(box)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["実機 (NI DAQ)", "シミュレーション"])
        form.addRow("モード", self.mode)
        self.daq_model = QtWidgets.QComboBox()
        self.daq_model.addItems(list(DAQ_MODELS))
        self.daq_model.setCurrentText("USB-6210")
        form.addRow("DAQ 機種", self.daq_model)
        self.device = QtWidgets.QLineEdit("Dev2")
        form.addRow("デバイス", self.device)
        self.durations = []
        for name, _, _ in PHASES:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(1, 600)
            sb.setValue(10)
            sb.setSuffix(" s")
            self.durations.append(sb)
            form.addRow(name, sb)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(3)
        self.threshold.setRange(0.5, 0.999)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(0.985)
        form.addRow("雲発生検出の I/I0 しきい値", self.threshold)
        self.r0 = QtWidgets.QDoubleSpinBox()
        self.r0.setRange(1e3, 1e5)
        self.r0.setDecimals(0)
        self.r0.setValue(cv.R0_TH)
        self.r0.setSuffix(" Ω")
        form.addRow("R0 (サーミスタ)", self.r0)
        self.data_dir = QtWidgets.QLineEdit(os.path.abspath(DEFAULT_DATA_DIR))
        form.addRow("データフォルダ", self.data_dir)
        self.cpc_port = QtWidgets.QLineEdit("COM3")
        self.cpc_port.setToolTip("CPC (TSI 3022A) のシリアルポート。空欄なら CPC を記録しない")
        form.addRow("CPC ポート", self.cpc_port)

        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("自動運転 開始")
        self.btn_manual = QtWidgets.QPushButton("手動運転 開始")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(lambda: self.start(manual=False))
        self.btn_manual.clicked.connect(lambda: self.start(manual=True))
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_manual)
        row.addWidget(self.btn_stop)
        v.addLayout(row)

        manual = QtWidgets.QGroupBox("手動運転: バルブ開閉操作")
        mrow = QtWidgets.QHBoxLayout(manual)
        self.btn_v1 = QtWidgets.QPushButton("V1（吸気）")
        self.btn_v2 = QtWidgets.QPushButton("V2（排気）")
        for btn in (self.btn_v1, self.btn_v2):
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.setStyleSheet("QPushButton:checked { background:#2ecc40; color:white; font-weight:bold; }")
            btn.toggled.connect(self._manual_valves)
            mrow.addWidget(btn)
        v.addWidget(manual)

        status = QtWidgets.QGroupBox("運転状態")
        g = QtWidgets.QGridLayout(status)
        g.setColumnStretch(1, 1)
        self.lbl_phase = QtWidgets.QLabel("停止中")
        self.lbl_phase.setStyleSheet("font-size: 13pt; font-weight: bold;")
        g.addWidget(self.lbl_phase, 0, 0, 1, 2)
        self.led_v1 = QtWidgets.QLabel("V1")
        self.led_v2 = QtWidgets.QLabel("V2")
        for led in (self.led_v1, self.led_v2):
            led.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            led.setFixedHeight(28)
        self._set_led(self.led_v1, None)
        self._set_led(self.led_v2, None)
        g.addWidget(self.led_v1, 1, 0)
        g.addWidget(self.led_v2, 1, 1)
        self.readouts = {}
        for i, (key, label) in enumerate([("pd", "PD_Vout [V]"), ("p", "フラスコ内の気圧 [hPa]"),
                                          ("T", "フラスコ内温度 [°C]"), ("rh", "露点計 RH [%]"),
                                          ("td", "露点計 露点温度 [°C]"),
                                          ("I0", "初期状態の I0 [V]"), ("tr", "I/I0"),
                                          ("cpc", "CPC [/cm³]")]):
            g.addWidget(QtWidgets.QLabel(label), 2 + i, 0)
            lab = QtWidgets.QLabel("—")
            lab.setWordWrap(True)  # long CPC error messages
            self.readouts[key] = lab
            g.addWidget(lab, 2 + i, 1)

        res = QtWidgets.QGroupBox("最新サイクルの推定値")
        rg = QtWidgets.QFormLayout(res)
        rg.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.lbl_time = QtWidgets.QLabel("—")
        self.lbl_qw = QtWidgets.QLabel("—")
        self.lbl_nd = QtWidgets.QLabel("—")
        self.lbl_flags = QtWidgets.QLabel("—")
        self.lbl_flags.setWordWrap(True)
        for lab in (self.lbl_qw, self.lbl_nd):
            lab.setStyleSheet("font-size: 12pt; font-weight: bold;")
        rg.addRow("時刻", self.lbl_time)
        rg.addRow("qw [g/kg]", self.lbl_qw)
        self.lbl_qwref = QtWidgets.QLabel("—")
        rg.addRow("qw 露点計 [g/kg]", self.lbl_qwref)
        self.lbl_td = QtWidgets.QLabel("—")
        self.lbl_td.setStyleSheet("font-size: 12pt; font-weight: bold;")
        rg.addRow("露点 [℃]", self.lbl_td)
        self.lbl_tdref = QtWidgets.QLabel("—")
        rg.addRow("露点 露点計 [℃]", self.lbl_tdref)
        rg.addRow("N [個/cm³]", self.lbl_nd)
        self.lbl_ndcpc = QtWidgets.QLabel("—")
        lab_cpc = QtWidgets.QLabel("N CPC [個/cm³]")
        lab_cpc.setToolTip("CPC の 1 サイクル（60 秒）の中央値（カウント数 ÷ 流量）")
        rg.addRow(lab_cpc, self.lbl_ndcpc)
        rg.addRow("フラグ", self.lbl_flags)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._scroll_tab(status, res), "運転状態・推定値")
        self.tabs.addTab(self._scroll_tab(box), "設定")
        v.addWidget(self.tabs, 1)
        return panel

    def _build_plots(self):
        w = pg.GraphicsLayoutWidget()
        pen = pg.mkPen(width=1.5)
        self.p_tr = w.addPlot(row=0, col=0, title="レーザー透過率 I/I0（直近60秒）")
        self.c_tr = self.p_tr.plot(pen=pen)
        self.p_tr.setLabel("bottom", "time", "s")
        self.p_p = w.addPlot(row=1, col=0, title="フラスコ内の気圧 [hPa]（直近60秒）")
        self.c_p = self.p_p.plot(pen=pen)
        self.p_p.setLabel("bottom", "time", "s")

        self.p_pfit = w.addPlot(row=0, col=1, title="直前の減圧: P(t) [hPa]")
        self.c_pdata = self.p_pfit.plot(pen=None, symbol="o", symbolSize=3)
        self.c_pfit = self.p_pfit.plot(pen=pg.mkPen("r", width=2))
        self.p_pfit.setLabel("bottom", "t − t0", "s")
        self.p_trfit = w.addPlot(row=1, col=1, title="直前の減圧: Tr(t)")
        self.c_trdata = self.p_trfit.plot(pen=pen)
        self.c_trfit = self.p_trfit.plot(pen=pg.mkPen("r", width=3))
        self.thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine))
        self.p_trfit.addItem(self.thr_line)
        self.p_trfit.setLabel("bottom", "t − t0", "s")

        axis = lambda: {"bottom": pg.DateAxisItem()}  # noqa: E731
        self.p_qw = w.addPlot(row=2, col=0, colspan=2, title="露点 [℃]",
                              axisItems=axis())
        self.p_qw.addLegend(offset=(10, 10))
        self.c_qw_ref = self.p_qw.plot(pen=None, symbol="t", symbolSize=6, symbolPen=None,
                                       symbolBrush=(150, 150, 150), name="露点計（参照）")
        self.c_qw = self.p_qw.plot(pen=None, symbol="o", symbolSize=5, name="雲チェンバー")
        self.e_qw = pg.ErrorBarItem(beam=0)
        self.p_qw.addItem(self.e_qw)
        self.p_nd = w.addPlot(row=3, col=0, colspan=2, title="エアロゾル数濃度 N [個/cm³]",
                              axisItems=axis())
        self.p_nd.addLegend(offset=(10, 10))
        self.c_nd_cpc = self.p_nd.plot(pen=None, symbol="t", symbolSize=6, symbolPen=None,
                                       symbolBrush=(150, 150, 150), name="CPC（1 サイクルの中央値、参照）")
        self.c_nd = self.p_nd.plot(pen=None, symbol="o", symbolSize=5, name="雲チェンバー")
        self.e_nd = pg.ErrorBarItem(beam=0)
        self.p_nd.addItem(self.e_nd)
        self.p_nd.setXLink(self.p_qw)
        return w

    @staticmethod
    def _set_led(led, state):
        color = {None: "#555", True: "#2ecc40", False: "#333"}[state]
        text = led.text().split()[0] + ("" if state is None else (" 開" if state else " 閉"))
        led.setText(text)
        led.setStyleSheet(f"background:{color}; color:white; border-radius:4px; font-weight:bold;")

    # ------------------------------------------------------------------ start / stop
    def start(self, manual=False):
        from .daq import SAFE_VALVES, NiDaq, SimulatedDaq
        from .estimators import LinearRegressionEstimator
        from .shape_estimator import ShapeSBIEstimator
        try:
            self.daq = (NiDaq(self.device.text().strip(), model=self.daq_model.currentText())
                        if self.mode.currentIndex() == 0 else SimulatedDaq())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DAQ エラー", f"デバイスを開けませんでした:\n{e}")
            return
        estimator = None
        if not manual:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                # Tr-shape SBI when a trained posterior exists (models/npe_shape2_*.pt, else npe_shape_*.pt)
                estimator = ShapeSBIEstimator()
            except Exception:
                estimator = LinearRegressionEstimator()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(f"推定器: {getattr(estimator, 'name', type(estimator).__name__)}")
        port = self.cpc_port.text().strip()
        if port:  # CPC logged to data/cpc/cpc_YYYYMMDD.csv; median with V1 closed also in cloud_data_*.csv
            from .cpc import CPCRecorder
            self.cpc = CPCRecorder(port, os.path.join(self.data_dir.text(), "cpc"), log=lambda s: None)
            self.cpc.start()
        self.ctrl = Controller(self.daq, estimator,
                               durations=[sb.value() for sb in self.durations],
                               tr_threshold=self.threshold.value(), r0=self.r0.value(),
                               data_dir=self.data_dir.text(), manual=manual, cpc=self.cpc)
        self.thr_line.setValue(self.threshold.value())
        # the controller opens the valves in the safe state; mirror it on the buttons
        for btn, state in zip((self.btn_v1, self.btn_v2), SAFE_VALVES):
            btn.blockSignals(True)
            btn.setChecked(state)
            btn.blockSignals(False)
            btn.setEnabled(manual)
        self.ctrl.start()
        self._enable_settings(False)
        self.tabs.setCurrentIndex(0)  # show the readouts

    def stop(self):
        if self.ctrl is not None:
            self.ctrl.stop()
            self.ctrl.join(5)
            self.ctrl = None
        if self.cpc is not None:
            self.cpc.stop()
            self.cpc.join(5)
            self.cpc = None
        self.lbl_phase.setText("停止中")
        self._set_led(self.led_v1, None)
        self._set_led(self.led_v2, None)
        for btn in (self.btn_v1, self.btn_v2):
            btn.setEnabled(False)
        self._enable_settings(True)

    def _manual_valves(self):
        if self.ctrl is not None and self.ctrl.manual:
            self.ctrl.request_valves(self.btn_v1.isChecked(), self.btn_v2.isChecked())

    def _enable_settings(self, on):
        for wdg in [self.mode, self.daq_model, self.device, self.threshold, self.r0, self.data_dir, self.cpc_port,
                    self.btn_start, self.btn_manual, *self.durations]:
            wdg.setEnabled(on)
        self.btn_stop.setEnabled(not on)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------ updates
    def _poll(self):
        if self.ctrl is None:
            return
        got_live = False
        while True:
            try:
                kind, payload = self.ctrl.out.get_nowait()
            except queue.Empty:
                break
            if kind == "live":
                self._push_live(payload)
                got_live = True
            elif kind == "phase":
                _, name, v1, v2 = payload
                self.lbl_phase.setText(name)
                self._set_led(self.led_v1, v1)
                self._set_led(self.led_v2, v2)
            elif kind == "cycle":
                self._show_cycle(payload)
            elif kind == "warning":  # one cycle could not be analysed; the run continues
                self.n_warn += 1
                self.statusBar().showMessage(
                    f"警告 {self.n_warn} 件（data フォルダの warnings_*.log を参照）: "
                    + payload.strip().splitlines()[-1])
            elif kind == "error":
                self.stop()
                QtWidgets.QMessageBox.critical(self, "エラー", payload)
                return
        if got_live:
            t = self.live["t"] - np.nanmax(self.live["t"])
            self.c_tr.setData(t, self.live["tr"], connect="finite")
            self.c_p.setData(t, self.live["p"] * 1e-2, connect="finite")

    def _push_live(self, d):
        n = d["t"].size
        for key in ("t", "tr", "p"):
            buf = self.live[key]
            buf[:-n] = buf[n:]
            buf[-n:] = d[key]
        self.readouts["pd"].setText(_fmt(d["pd"][-1]))
        self.readouts["p"].setText(_fmt(d["p"][-1] * 1e-2, "{:.1f}"))
        self.readouts["T"].setText(_fmt(d["T"][-1] - 273.15, "{:.2f}"))
        self.readouts["rh"].setText(_fmt(d["rh"][-1] * 100, "{:.1f}"))
        self.readouts["td"].setText(_fmt(d["td"][-1], "{:.2f}"))
        self.readouts["tr"].setText(_fmt(d["tr"][-1], "{:.4f}"))
        self.readouts["I0"].setText(_fmt(self.ctrl.I0, "{:.4f}") if self.ctrl else "—")
        if self.cpc is None:
            self.readouts["cpc"].setText("—（未接続）")
        elif self.cpc.error:
            self.readouts["cpc"].setText("エラー: " + self.cpc.error[:40])
        else:
            self.readouts["cpc"].setText(_fmt(self.cpc.latest()[1], "{:.0f}"))

    def _show_cycle(self, c):
        self.cycles = (self.cycles + [c])[-HISTORY_CYCLES:]
        self.lbl_time.setText(f"{datetime.fromtimestamp(c['t_wall']):%Y-%m-%d %H:%M:%S}")
        self.lbl_qw.setText(self._with_ci(c.get("qw_gkg"), c.get("qw_lo"), c.get("qw_hi"), "{:.2f}"))
        self.lbl_nd.setText(self._with_ci(c.get("nd_cm3"), c.get("nd_cm3_lo"),
                                          c.get("nd_cm3_hi"), "{:.0f}"))
        self.lbl_ndcpc.setText(_fmt(c.get("cpc_cm3", np.nan), "{:.0f}"))
        self.lbl_flags.setText(", ".join(c["flags"]) or "なし")
        self.lbl_qwref.setText(_fmt(self._qw_ref(c), "{:.2f}"))
        self.lbl_td.setText(self._with_ci(c.get("Td_est"), c.get("Td_est_lo"), c.get("Td_est_hi"),
                                          "{:.2f}"))
        self.lbl_tdref.setText(_fmt(c.get("ref_Td", np.nan), "{:.2f}"))

        # time axis from the pressure onset; ramp + double-exponential fit when the Tr-shape
        # estimator ran, else the single exponential of Eq. (111)
        if np.isfinite(c.get("t0_ramp2", np.nan)):
            t0 = c["t0_ramp2"]
            tt = c["t"] - t0
            pf = pressure_ramp2(c["t"], c["p_ini"], t0, c["p_offset2"], c["tau_vac1"], c["frac1"],
                                c["tau_vac2"], c["t_open"])
            self.c_pfit.setData(tt, pf * 1e-2)
        else:
            t0 = c["t0"] if np.isfinite(c["t0"]) else 0.0
            tt = c["t"] - t0
            if np.isfinite(c["tau_vac"]):
                tf = np.linspace(0, tt[-1], 200)
                pf = (c["p_ini"] - c["p_offset"]) * np.exp(-tf / c["tau_vac"]) + c["p_offset"]
                self.c_pfit.setData(tf, pf * 1e-2)
            else:
                self.c_pfit.setData([], [])
        self.c_pdata.setData(tt, c["p"] * 1e-2)
        self.c_trdata.setData(tt, c["tr"], connect="finite")
        t_s = c.get("t_s_shape", np.nan)
        if np.isfinite(t_s):  # red: the part of Tr(t) used by the Tr-shape estimator
            offsets = getattr(self.ctrl.estimator if self.ctrl else None, "offsets", SHAPE_OFFSETS)
            seg = (tt >= t_s + offsets[0]) & (tt <= t_s + offsets[-1])
            self.c_trfit.setData(tt[seg], c["tr"][seg])
        else:
            self.c_trfit.setData([], [])

        ts = np.array([x["t_wall"] for x in self.cycles])
        self.c_qw_ref.setData(ts, np.array([x.get("ref_Td", np.nan) for x in self.cycles]),
                              connect="finite")
        self.c_nd_cpc.setData(ts, np.array([x.get("cpc_cm3", np.nan) for x in self.cycles]),
                              connect="finite")
        for curve, err, key in ((self.c_qw, self.e_qw, "Td_est"), (self.c_nd, self.e_nd, "nd_cm3")):
            mid = np.array([x.get(key, np.nan) for x in self.cycles])
            lo = np.array([x.get(f"{key}_lo", np.nan) for x in self.cycles])
            hi = np.array([x.get(f"{key}_hi", np.nan) for x in self.cycles])
            curve.setData(ts, mid, connect="finite")
            ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(mid)
            if ok.any():
                err.setData(x=ts[ok], y=mid[ok], top=hi[ok] - mid[ok], bottom=mid[ok] - lo[ok])

    @staticmethod
    def _qw_ref(c):
        """Reference qw [g/kg]: from the dew point when available, else from the RH output."""
        q = c.get("qw_ref_td_gkg", np.nan)
        return q if np.isfinite(q) else c.get("qw_ref_gkg", np.nan)

    @staticmethod
    def _with_ci(mid, lo, hi, f):
        if mid is None or not np.isfinite(mid):
            return "—"
        s = f.format(mid)
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            s += f"  [{f.format(lo)}, {f.format(hi)}]"
        return s


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 950)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
