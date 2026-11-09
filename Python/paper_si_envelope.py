# -*- coding: utf-8 -*-
"""SI figure: operating envelope of the chamber from the forward model.

For the recorded pressure curve of the high-N cycle of fig_cycle (p_ini kept), the forward model is
run on a grid of water vapor mixing ratio qw (up to saturation at T_ini) and number concentration N
for three initial temperatures T_INIS, with nominal nuisance parameters. For each run it reports
  - whether a measurable cloud forms (Tr falls to 0.985 within the simulated T_END),
  - the peak supersaturation S_max - 1,
  - the smallest activated dry diameter for kappa = 0.1 at the peak (kappa-Koehler, as in
    paper_si_cutoff.py).
Hatched: no measurable cloud. Thick line: S_max - 1 = 300 % (saturation ratio 4, above which drops
form also in particle-free air).
The evaluated range of the paper (qw 5.8-9.2 g/kg, N 370-9500 cm-3, T_ini about 296 K) is marked.
The lowest N that gives a measurable cloud at 23 C is also found on a finer N grid for the qw of
QW_SCAN (the numbers quoted in SI Sect. S10).

Usage: python paper_si_envelope.py  -> Paper/JAerosolSci/figures/fig_s_envelope.{pdf,png}
"""
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps
from paper_fig_sensitivity import pressure_curve, run
from paper_si_cutoff import cutoff_diameter
from paper_style import INK, tidy

T_INIS = (283.15, 296.15, 303.15)  # [K]
N_GRID = np.logspace(np.log10(50.0), np.log10(3e4), 37)  # [cm-3]
N_QW = 40
QW_SCAN = (2.0, 3.0, 4.0, 6.0, 9.0)  # [g/kg] lowest N with a measurable cloud at 23 C
N_SCAN = np.logspace(1.5, 4.0, 51)  # [cm-3]
SI_WIDTH = 165 * ps.MM  # [inch] text width of the supplement (article, 25-mm margins)
LABEL_X = 0.8  # contour labels on the column at this fraction of the qw axis (clear of the marked box)
KAPPA = 0.1
TR_ONSET = 0.985
EVAL = dict(qw=(5.8, 9.2), n=(370.0, 9500.0))


def qw_sat(T, p):
    """Saturation mixing ratio [g/kg] over water at T [K] and p [Pa] (Bolton 1980)."""
    e = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 622.0 * e / (p - e)


def one(args):
    qw, n, cond = args
    s = run(qw, n, cond)
    i = int(np.argmax(s["S"]))
    cloud = bool(np.min(s["Tr"]) <= TR_ONSET)
    return s["S"][i] - 1.0, cutoff_diameter(s["S"][i], s["T"][i], KAPPA), cloud


def cloud_only(args):
    qw, n, cond = args
    return bool(np.min(run(qw, n, cond)["Tr"]) <= TR_ONSET)


def label_points(qws, zz, levels):
    """Positions (qw, N) where the contours of zz (shape (N_QW, N)) cross the column at LABEL_X."""
    j = int(np.argmin(np.abs(qws - (qws[0] + LABEL_X * (qws[-1] - qws[0])))))
    col, lg = zz[j], np.log10(N_GRID)
    pts = []
    for lev in levels:
        for i in range(len(col) - 1):
            a, b = col[i], col[i + 1]
            if np.isfinite(a) and np.isfinite(b) and (a - lev) * (b - lev) < 0:
                y = 10 ** (lg[i] + (lev - a) / (b - a) * (lg[i + 1] - lg[i]))
                if N_GRID[1] < y < N_GRID[-3]:
                    pts.append((qws[j], y))
                break
    return pts


def main():
    base = pressure_curve()
    cond23 = dict(base, T_ini=296.15)
    with Pool() as pool:
        ok = np.array(pool.map(cloud_only, [(q, n, cond23) for q in QW_SCAN for n in N_SCAN])).reshape(
            len(QW_SCAN), len(N_SCAN))
    for q, o in zip(QW_SCAN, ok):
        print(f"T_ini 23 C, qw {q:.0f} g/kg: lowest N with a measurable cloud "
              f"{N_SCAN[o].min() if o.any() else np.nan:.0f} cm-3 (grid step {N_SCAN[1] / N_SCAN[0]:.2f}x)")
    results = []
    for T in T_INIS:
        cond = dict(base, T_ini=T)
        qs = qw_sat(T, cond["p_ini"])
        qws = np.linspace(2.0, 0.98 * qs, N_QW)
        with Pool() as pool:
            out = pool.map(one, [(q, n, cond) for q in qws for n in N_GRID])
        smax, dcut, cloud = (np.array(v, dtype=float).reshape(N_QW, len(N_GRID)) for v in zip(*out))
        results.append((T, qws, smax, dcut, cloud))
        nc = np.where(cloud.astype(bool), 0, 1)
        print(f"T_ini {T - 273.15:.0f} C: qw 2-{qws[-1]:.1f} g/kg; S_max-1 {np.nanmin(smax)*100:.0f}-"
              f"{np.nanmax(smax)*100:.0f} %; no measurable cloud in {nc.mean()*100:.0f} % of the grid; "
              f"cutoff (kappa {KAPPA}) {np.nanmin(dcut):.1f}-{np.nanmax(dcut):.1f} nm")
        for q_lo in (2.0, 3.0, 4.0):
            j = int(np.argmin(np.abs(qws - q_lo)))
            print(f"   qw {qws[j]:.1f}: cloud for N >= "
                  f"{N_GRID[cloud[j].astype(bool)].min() if cloud[j].any() else np.nan:.0f} cm-3, "
                  f"S_max-1 {smax[j].min()*100:.0f}-{smax[j].max()*100:.0f} %, cutoff {dcut[j].min():.1f}-{dcut[j].max():.1f} nm")
        wilson = (smax > 3.0) & cloud.astype(bool) & (qws[:, None] > 5.0)
        if wilson.any():
            print(f"   S_max > 4 for qw > 5 g/kg up to N = {N_GRID[wilson.any(axis=0)].max():.0f} cm-3")
        big = cloud.astype(bool) & (N_GRID[None, :] <= 1e4 * 1.0001)
        print(f"   cutoff (kappa {KAPPA}) up to 1e4 cm-3: max {np.nanmax(np.where(big, dcut, np.nan)):.1f} nm; "
              f"at the highest N {np.nanmin(np.where(cloud[:, -1:].astype(bool), dcut[:, -1:], np.nan)):.1f}-"
              f"{np.nanmax(np.where(cloud[:, -1:].astype(bool), dcut[:, -1:], np.nan)):.1f} nm (cloud only)")

    ps.apply()
    # SI text width (about 165 mm): one row per T_ini; left column peak supersaturation, right column cutoff
    fig, axs = plt.subplots(len(T_INIS), 2, figsize=(SI_WIDTH, 190 * ps.MM), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.14, top=0.965, hspace=0.42, wspace=0.12)
    lev_s = [5, 10, 20, 40, 60, 80, 120, 160, 200, 300]
    lev_d = [2, 3, 4, 5, 7, 10, 14, 20]
    for j, (T, qws, smax, dcut, cloud) in enumerate(results):
        for i, (z, lev, cmap) in enumerate(((smax * 100, lev_s, "Blues"), (dcut, lev_d, "Oranges_r"))):
            ax = axs[j, i]
            zz = np.where(cloud.astype(bool), z, np.nan)
            cf = ax.contourf(qws, N_GRID, zz.T, levels=lev, cmap=cmap, extend="both")
            cs = ax.contour(qws, N_GRID, zz.T, levels=lev, colors=INK, linewidths=0.4)
            shown = [v for v in lev if not (i == 0 and v == 300)]  # 300 % is the thick line
            ax.clabel(cs, levels=shown, fontsize=6, fmt="%g", manual=label_points(qws, zz, shown))
            ax.contourf(qws, N_GRID, cloud.T.astype(float), levels=[-0.5, 0.5], colors="none", hatches=["////"])
            if i == 0 and np.nanmax(zz) > 300:  # S > 4: drops also form without particles (Wilson)
                ax.contour(qws, N_GRID, zz.T, levels=[300], colors=INK, linewidths=1.6)
            if abs(T - 296.15) < 1:
                ax.add_patch(plt.Rectangle((EVAL["qw"][0], EVAL["n"][0]), np.diff(EVAL["qw"])[0],
                                           np.diff(EVAL["n"])[0], fill=False, ec=INK, lw=1.2, ls="--"))
            ax.set_yscale("log")
            tidy(ax)
            ax.set_title(f"$\\mathit{{T}}_\\mathrm{{ini}}$ = {T - 273.15:.0f} °C", fontsize=8)
            ax.set_xlabel(r"$\mathit{q}_\mathit{w}$ (g kg$^{-1}$)")
            if i == 0:
                ax.set_ylabel(r"$\mathit{N}$ (cm$^{-3}$)")
            ps.panel(ax, "abcdef"[i * len(T_INIS) + j], x=-0.2 if i == 0 else -0.1)
        if j == len(T_INIS) - 1:
            for i in range(2):
                pos = axs[j, i].get_position()
                cax = fig.add_axes([pos.x0, 0.055, pos.width, 0.012])
                cb = fig.colorbar(axs[j, i].collections[0], cax=cax, orientation="horizontal")
                cb.set_label("Peak supersaturation (%)" if i == 0 else
                             f"Cutoff dry diameter, $\\mathit{{\\kappa}}$ = {KAPPA} (nm)")
    ps.save(fig, "fig_s_envelope")


if __name__ == "__main__":
    main()
