#!/usr/bin/env python
"""Velocity profile of a Carreau-Yasuda fluid in a circular tube at a given pressure gradient.

Examples
--------
Paper fluid at G = 0.6 MPa/m in a 1 cm tube (the conditions of Fig. 3):

    python scripts/velocity_profile.py --G 6e5

Your own fluid, plus the Fig. 3(a)-style plot of the terms of Eq. (22):

    python scripts/velocity_profile.py --eta0 50 --eta-inf 0.1 --lam 0.5 --n 0.4 --a 2 \
        --radius 0.002 --G 2e6 --terms --csv profile.csv

The figure shows u(r) across the full diameter (with the Newtonian profile
of equal flow rate for comparison), the local shear rate and the local
viscosity.  Integral quantities are printed to the terminal.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _cli import add_fluid_arguments, fluid_from_args, fluid_label
from cy_pipeflow import velocity_profile, reduced_parameters, r_tilde, w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_fluid_arguments(p)
    p.add_argument("--G", type=float, required=True, help="pressure drop per unit length [Pa/m]")
    p.add_argument("--points", type=int, default=400, help="number of points in s")
    p.add_argument("--terms", action="store_true",
                   help="add a panel with 1 - s r(s), w(s), r(s) vs s (Fig. 3(a) style)")
    p.add_argument("--out", default="velocity_profile.png", help="output figure")
    p.add_argument("--csv", default=None, help="optional CSV output of the profile")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    fluid = fluid_from_args(args)
    R, G = args.radius, args.G
    prof = velocity_profile(G, R, fluid, n_points=args.points)
    res = prof.result

    print(f"G = {G:g} Pa/m, R = {R:g} m, fluid = {fluid}")
    print(f"  wall shear stress   tau_wall   = {res.tau_wall:.6g} Pa")
    print(f"  wall shear rate     gamma_wall = {res.gamma_wall:.6g} 1/s")
    print(f"  wall viscosity      eta_wall   = {res.eta_wall:.6g} Pa s")
    print(f"  centreline velocity u_max      = {res.u_max:.6g} m/s")
    print(f"  average velocity    u_avg      = {res.u_avg:.6g} m/s")
    print(f"  volume flow rate    Q          = {res.Q:.6g} m^3/s  ({res.Q*1e6:.6g} mL/s)")
    print(f"  u_max/u_avg = {res.ratio:.5f}   (2 Newtonian, {fluid.power_law_ratio:.4f} power-law)")
    print(f"  reduced: u_max_tilde = {res.u_max_tilde:.6f}, u_avg_tilde = {res.u_avg_tilde:.6f}")

    if args.csv:
        np.savetxt(args.csv,
                   np.column_stack([prof.s, prof.r, prof.u, prof.gamma, prof.eta, prof.tau,
                                    prof.r_tilde, prof.u_tilde]),
                   delimiter=",", comments="", fmt="%.10g",
                   header="s,r[m],u[m/s],gamma_dot[1/s],eta[Pa s],tau[Pa],r_tilde,u_tilde")
        print(f"Wrote {args.csv}")

    ncol = 4 if args.terms else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2))
    ax_u, ax_g, ax_e = axes[:3]
    rmm = prof.r * 1e3

    # full-diameter velocity profile and the Newtonian parabola of equal Q
    ax_u.plot(np.r_[-rmm[::-1], rmm], np.r_[prof.u[::-1], prof.u], "-", color="firebrick",
              lw=2.2, label="Carreau–Yasuda, Eq. (22)")
    rr = np.linspace(-R, R, 201)
    ax_u.plot(rr * 1e3, 2 * res.u_avg * (1 - (rr / R) ** 2), "--", color="grey", lw=1.3,
              label="Newtonian, same Q")
    ax_u.set_xlabel(r"$r$  [mm]")
    ax_u.set_ylabel(r"$u(r)$  [m/s]")
    ax_u.set_ylim(bottom=0)
    ax_u.legend(frameon=False, fontsize=8)

    ax_g.plot(rmm, prof.gamma, color="navy", lw=2)
    ax_g.set_xlabel(r"$r$  [mm]")
    ax_g.set_ylabel(r"$\dot\gamma(r)$  [s$^{-1}$]")
    ax_g.set_ylim(bottom=0)

    ax_e.semilogy(rmm, prof.eta, color="darkgreen", lw=2)
    ax_e.set_xlabel(r"$r$  [mm]")
    ax_e.set_ylabel(r"$\eta(r)$  [Pa·s]")

    if args.terms:
        rp = reduced_parameters(G, R, fluid)
        s = prof.s
        rt = r_tilde(s, rp, fluid)
        ws = w(s, rp, fluid)
        ax_t = axes[3]
        ax_t.plot(s, 1 - s * rt, "k-", lw=1.8, label=r"$1 - s\,\tilde r(s)$")
        ax_t.plot(s, ws, "b-", lw=1.8, label=r"$w(s)$")
        ax_t.plot(s, rt, "r-", lw=1.8, label=r"$\tilde r(s)$")
        ax_t.fill_between(s, ws, 1 - s * rt, color="lightblue", alpha=0.6)
        ax_t.set_xlabel(r"$s = \dot\gamma/\dot\gamma_{\rm wall}$")
        ax_t.set_ylim(0, 1.02)
        ax_t.legend(frameon=False, fontsize=8)
        ax_t.set_title("terms of Eq. (22)", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"G = {G/1e6:g} MPa/m, R = {R*1e3:g} mm, "
                 rf"$\dot\gamma_{{\rm wall}}$ = {res.gamma_wall:.4g} s$^{{-1}}$, "
                 f"Q = {res.Q*1e6:.4g} mL/s, $u_{{\\max}}/u_{{\\rm avg}}$ = {res.ratio:.3f}\n"
                 f"{fluid_label(fluid)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
