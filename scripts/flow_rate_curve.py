#!/usr/bin/env python
"""Volume flow rate vs pressure gradient for a Carreau-Yasuda fluid.

Examples
--------
Paper fluid, R = 1 cm, G from 0.005 to 1 MPa/m (log-spaced), plot + CSV:

    python scripts/flow_rate_curve.py --gmin 5e3 --gmax 1e6 --log \
        --out Q_vs_G.png --csv Q_vs_G.csv

Your own fluid:

    python scripts/flow_rate_curve.py --eta0 50 --eta-inf 0.1 --lam 0.5 --n 0.4 --a 2 \
        --radius 0.002 --gmin 1e4 --gmax 1e7 --num 80 --log

The figure shows Q, u_avg, the wall shear rate and u_max/u_avg against G.
The CSV contains every column of the flow_rate_curve() dict.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _cli import add_fluid_arguments, fluid_from_args, fluid_label
from cy_pipeflow import flow_rate_curve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_fluid_arguments(p)
    p.add_argument("--gmin", type=float, default=5e3, help="smallest pressure gradient [Pa/m]")
    p.add_argument("--gmax", type=float, default=1e6, help="largest pressure gradient [Pa/m]")
    p.add_argument("--num", type=int, default=60, help="number of G values")
    p.add_argument("--log", action="store_true", help="log-spaced G values and log-log axes")
    p.add_argument("--out", default="flow_rate_curve.png", help="output figure")
    p.add_argument("--csv", default=None, help="optional CSV output")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    fluid = fluid_from_args(args)
    R = args.radius
    if args.log:
        G = np.logspace(np.log10(args.gmin), np.log10(args.gmax), args.num)
    else:
        G = np.linspace(args.gmin, args.gmax, args.num)

    c = flow_rate_curve(G, R, fluid)

    if args.csv:
        cols = ["G", "Q", "u_avg", "u_max", "gamma_wall", "eta_wall", "tau_wall", "ratio",
                "u_max_tilde", "u_avg_tilde"]
        header = ("G[Pa/m],Q[m^3/s],u_avg[m/s],u_max[m/s],gamma_wall[1/s],eta_wall[Pa s],"
                  "tau_wall[Pa],u_max/u_avg,u_max_tilde,u_avg_tilde")
        np.savetxt(args.csv, np.column_stack([c[k] for k in cols]), delimiter=",",
                   header=header, comments="", fmt="%.10g")
        print(f"Wrote {args.csv}")

    plot = (lambda ax, x, y, **kw: ax.loglog(x, y, **kw)) if args.log \
        else (lambda ax, x, y, **kw: ax.plot(x, y, **kw))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    (ax1, ax2), (ax3, ax4) = axes
    Gm = c["G"] / 1e6

    plot(ax1, Gm, c["Q"] * 1e6, color="firebrick", lw=2)
    ax1.set_ylabel(r"$Q$  [mL/s]")
    plot(ax2, Gm, c["u_avg"], color="navy", lw=2)
    ax2.set_ylabel(r"$u_{\mathrm{avg}}$  [m/s]")
    plot(ax3, Gm, c["gamma_wall"], color="darkgreen", lw=2)
    ax3.set_ylabel(r"$\dot\gamma_{\mathrm{wall}}$  [s$^{-1}$]")
    if args.log:
        ax4.semilogx(Gm, c["ratio"], color="darkorange", lw=2)
    else:
        ax4.plot(Gm, c["ratio"], color="darkorange", lw=2)
    ax4.axhline(2.0, color="grey", ls=":", lw=1)
    ax4.axhline(fluid.power_law_ratio, color="grey", ls=":", lw=1)
    ax4.set_ylabel(r"$u_{\max}/u_{\mathrm{avg}}$")
    ax4.text(0.02, 0.93, "2 = Newtonian", transform=ax4.transAxes, fontsize=8, color="grey")
    ax4.text(0.02, 0.05, f"(3n+1)/(n+1) = {fluid.power_law_ratio:.3f}",
             transform=ax4.transAxes, fontsize=8, color="grey")
    for ax in axes.flat:
        ax.set_xlabel(r"$G$  [MPa/m]")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Carreau–Yasuda pipe flow, R = {R*1e3:g} mm\n{fluid_label(fluid)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Wrote {args.out}")
    print(f"Q ranges from {c['Q'][0]:.4e} to {c['Q'][-1]:.4e} m^3/s "
          f"over G = {args.gmin:g} .. {args.gmax:g} Pa/m")


if __name__ == "__main__":
    main()
