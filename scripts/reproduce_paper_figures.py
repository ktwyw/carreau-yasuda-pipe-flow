#!/usr/bin/env python
"""Reproduce Figs. 1-3 of Wang, JNNFM 310 (2022) 104937.

    python scripts/reproduce_paper_figures.py [--outdir figures] [--dpi 200]

Fig. 1  viscosity and shear stress vs shear rate (example fluid)
Fig. 2  (a) u_avg vs G, (b) u_max/u_avg vs G  -- lines: Eqs. (23),(31);
        points: independent quadrature of Eqs. (17),(26)
Fig. 3  G = 0.6 MPa/m, R = 1 cm: (a) the terms of Eq. (22) vs s,
        (b) analytical vs numerical velocity distribution
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cy_pipeflow import (EXAMPLE_FLUID, reduced_parameters, flow_rate_curve,
                         r_tilde, w, u_tilde)
from cy_pipeflow.numerical import (u_avg_tilde_numerical, u_max_tilde_numerical,
                                   u_tilde_numerical)


def make_fig1(fluid, fname, dpi):
    gamma = np.logspace(-2.5, 3.5, 400)
    eta = fluid.viscosity(gamma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax1.loglog(gamma, eta, "k-", lw=1.8)
    ax1.set_xlabel(r"$\dot\gamma$  [s$^{-1}$]")
    ax1.set_ylabel(r"$\eta(\dot\gamma)$  [Pa$\cdot$s]")
    ax1.set_title("(a)")
    ax2.loglog(gamma, eta * gamma, "k-", lw=1.8)
    ax2.set_xlabel(r"$\dot\gamma$  [s$^{-1}$]")
    ax2.set_ylabel(r"$\eta(\dot\gamma)\,\dot\gamma$  [Pa]")
    ax2.set_title("(b)")
    fig.suptitle("Fig. 1 — Carreau–Yasuda fluid: viscosity and shear stress vs shear rate")
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def make_fig2(R, fluid, fname, dpi):
    G_line = np.linspace(0.005e6, 1.0e6, 160)
    G_pts = np.linspace(0.05e6, 1.0e6, 20)

    line = flow_rate_curve(G_line, R, fluid)                 # analytical

    ua_pts, um_pts = [], []                                  # independent check
    for G in G_pts:
        rp = reduced_parameters(G, R, fluid)
        scale = R * rp.gamma_wall
        ua_pts.append(scale * u_avg_tilde_numerical(rp, fluid))
        um_pts.append(scale * u_max_tilde_numerical(rp, fluid))
    ua_pts, um_pts = np.array(ua_pts), np.array(um_pts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax1.semilogy(G_line / 1e6, line["u_avg"], "-", color="lightcoral", lw=2,
                 label="analytical, Eq. (31)")
    ax1.semilogy(G_pts / 1e6, ua_pts, "o", color="mediumblue", ms=5,
                 label="numerical, Eq. (26)")
    ax1.set_xlabel(r"$G$  [MPa/m]")
    ax1.set_ylabel(r"$u_{\mathrm{avg}}$  [m/s]")
    ax1.set_title("(a)")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(G_line / 1e6, line["ratio"], "-", color="cyan", lw=2, label="analytical")
    ax2.plot(G_pts / 1e6, um_pts / ua_pts, "o", color="mediumblue", ms=5, label="numerical")
    ax2.set_xlabel(r"$G$  [MPa/m]")
    ax2.set_ylabel(r"$u_{\mathrm{max}} / u_{\mathrm{avg}}$")
    ax2.set_ylim(1.2, 2.0)
    ax2.set_title("(b)")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig. 2 — Average velocity and velocity ratio vs pressure gradient")
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    pts = flow_rate_curve(G_pts, R, fluid)
    return line["ratio"][0], line["ratio"][-1], np.max(np.abs(um_pts / ua_pts - pts["ratio"]))


def make_fig3(G, R, fluid, fname, dpi):
    rp = reduced_parameters(G, R, fluid)
    s = np.linspace(0.0, 1.0, 400)
    rt = r_tilde(s, rp, fluid)
    ws = w(s, rp, fluid)
    one_minus_srt = 1.0 - s * rt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax1.plot(s, one_minus_srt, "k-", lw=1.8, label=r"$1 - s\,\tilde r(s)$")
    ax1.plot(s, ws, "b-", lw=1.8, label=r"$w(s)$")
    ax1.plot(s, rt, "r-", lw=1.8, label=r"$\tilde r(s)$")
    ax1.fill_between(s, ws, one_minus_srt, color="lightblue", alpha=0.6)
    ax1.set_xlabel(r"$s$")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("(a)")
    ax1.legend(frameon=False)

    ut_an = u_tilde(s, rp, fluid)
    ut_num = u_tilde_numerical(s, rp, fluid)
    ax2.plot(rt, ut_an, "r-", lw=2.0, label="analytical, Eq. (22)")
    ax2.plot(rt, ut_num, "b--", lw=1.4, label="numerical, Eq. (17)")
    ax2.set_xlabel(r"$\tilde r$")
    ax2.set_ylabel(r"$\tilde u$")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.20)
    ax2.set_title("(b)")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(rf"Fig. 3 — Parametric solution, $G$ = {G/1e6:.1f} MPa/m, "
                 rf"$R$ = {R*100:.0f} cm  ($\dot\gamma_{{\rm wall}}$ = {rp.gamma_wall:.1f} s$^{{-1}}$)")
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    return rp.gamma_wall, ut_an[0], np.max(np.abs(ut_an - ut_num))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fluid, R = EXAMPLE_FLUID, 0.01
    make_fig1(fluid, os.path.join(args.outdir, "fig1.png"), args.dpi)
    lo, hi, dev2 = make_fig2(R, fluid, os.path.join(args.outdir, "fig2.png"), args.dpi)
    gw, umt, dev3 = make_fig3(0.6e6, R, fluid, os.path.join(args.outdir, "fig3.png"), args.dpi)

    print(f"Figures written to {args.outdir}/: fig1.png fig2.png fig3.png\n")
    print(f"u_max/u_avg at G = 0.005 MPa/m : {lo:.4f}   (Newtonian limit 2)")
    print(f"u_max/u_avg at G = 1.0   MPa/m : {hi:.4f}   (power-law limit {fluid.power_law_ratio:.4f})")
    print(f"max |analytical - numerical| in Fig. 2(b) : {dev2:.1e}")
    print(f"\nFig. 3: wall shear rate = {gw:.2f} 1/s, reduced u_max = {umt:.6f}")
    print(f"max |analytical - numerical| in Fig. 3(b) : {dev3:.1e}")


if __name__ == "__main__":
    main()
