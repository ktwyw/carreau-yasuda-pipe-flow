"""Shared command-line options for the scripts (fluid parameters and radius)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cy_pipeflow import CarreauYasuda, EXAMPLE_FLUID  # noqa: E402


def add_fluid_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group(
        "Carreau-Yasuda parameters (defaults: the example fluid of the paper)")
    g.add_argument("--eta0", type=float, default=EXAMPLE_FLUID.eta0,
                   help="zero-shear-rate viscosity [Pa s]")
    g.add_argument("--eta-inf", type=float, default=EXAMPLE_FLUID.eta_inf,
                   help="infinite-shear-rate viscosity [Pa s]")
    g.add_argument("--lam", type=float, default=EXAMPLE_FLUID.lam,
                   help="time constant lambda [s]")
    g.add_argument("--n", type=float, default=EXAMPLE_FLUID.n,
                   help="power-law exponent [-]")
    g.add_argument("--a", type=float, default=EXAMPLE_FLUID.a,
                   help="Yasuda transition parameter [-]")
    parser.add_argument("--radius", "-R", type=float, default=0.01,
                        help="tube radius [m] (default 0.01)")


def fluid_from_args(args) -> CarreauYasuda:
    return CarreauYasuda(eta0=args.eta0, eta_inf=args.eta_inf,
                         lam=args.lam, n=args.n, a=args.a)


def fluid_label(f: CarreauYasuda) -> str:
    return (rf"$\eta_0$={f.eta0:g} Pa·s, $\eta_\infty$={f.eta_inf:g} Pa·s, "
            rf"$\lambda$={f.lam:g} s, $n$={f.n:g}, $a$={f.a:g}")
