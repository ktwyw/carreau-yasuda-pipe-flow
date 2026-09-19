#!/usr/bin/env python
"""Animated 3-D 'dome' of the velocity profile as the pressure gradient sweeps up.

    python scripts/dome_animation.py                 # u/(R gamma_wall), fixed z-scale 0..0.5
    python scripts/dome_animation.py --normalized    # u/u_max: peak pinned at 1, shape only

The un-normalised version shows both the flattening and the drop of the
reduced centreline velocity from 0.5 (Newtonian) to (n+1)/(3n+1)-like
plug flow; the normalised version isolates the change of SHAPE.  The GIF
loops and drops straight into a slide deck.  Any fluid parameters and radius
can be given with the same options as the other scripts.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

from _cli import add_fluid_arguments, fluid_from_args
from cy_pipeflow import velocity_profile


def revolve(r_t, u_t, n_theta=80):
    """Surface of revolution (X, Y, Z) of a profile about the tube axis."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    T, Rm = np.meshgrid(theta, r_t)
    _, Um = np.meshgrid(theta, u_t)
    return Rm * np.cos(T), Rm * np.sin(T), Um


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_fluid_arguments(p)
    p.add_argument("--gmin", type=float, default=0.02e6, help="first G [Pa/m]")
    p.add_argument("--gmax", type=float, default=1.0e6, help="last G [Pa/m]")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--normalized", action="store_true", help="plot u/u_max instead of u/(R gamma_wall)")
    p.add_argument("--out", default=None, help="output GIF (default depends on --normalized)")
    args = p.parse_args()

    fluid, R = fluid_from_args(args), args.radius
    fname = args.out or ("dome_animation_normalized.gif" if args.normalized else "dome_animation.gif")
    G_values = np.logspace(np.log10(args.gmin), np.log10(args.gmax), args.frames)
    n_lim = (fluid.n + 1.0) / (3.0 * fluid.n + 1.0)   # u_avg/u_max, power-law limit
    zmax = 1.0 if args.normalized else 0.5

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(projection="3d")

    def draw(i):
        ax.clear()
        G = G_values[i]
        prof = velocity_profile(G, R, fluid, n_points=140)
        res = prof.result
        z = prof.u_tilde / prof.u_tilde[0] if args.normalized else prof.u_tilde
        X, Y, Z = revolve(prof.r_tilde, z)
        ax.plot_surface(X, Y, Z, facecolors=cm.rainbow(Z / zmax), rstride=6, cstride=3,
                        linewidth=0.2, edgecolor=(0, 0, 0, 0.3), antialiased=True, shade=False)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, zmax)
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev=22, azim=-60)
        ax.set_xlabel(r"$x/R$"); ax.set_ylabel(r"$y/R$")
        head = (f"$G$ = {G/1e6:.3f} MPa/m    "
                rf"$\dot\gamma_{{\rm wall}}$ = {res.gamma_wall:.3g} s$^{{-1}}$")
        if args.normalized:
            ax.set_zlabel(r"$u/u_{\max}$")
            ax.set_title(head + "\n"
                         rf"$u_{{\rm avg}}/u_{{\max}}$ = {res.u_avg/res.u_max:.3f}"
                         rf"   (0.5 Newtonian $\rightarrow$ {n_lim:.2f} power-law)")
        else:
            ax.set_zlabel(r"$\tilde u$")
            ax.set_title(head + rf"    $\tilde u_{{\max}}$ = {res.u_max_tilde:.3f}")

    anim = FuncAnimation(fig, draw, frames=args.frames)
    anim.save(fname, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
