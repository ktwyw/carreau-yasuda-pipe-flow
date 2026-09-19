"""cy_pipeflow -- analytical Hagen-Poiseuille flow of a Carreau-Yasuda fluid.

Implements Y. Wang, J. Non-Newtonian Fluid Mech. 310 (2022) 104937.

Quick start
-----------
>>> from cy_pipeflow import CarreauYasuda, solve_pipe_flow, velocity_profile
>>> fluid = CarreauYasuda(eta0=1400.0, eta_inf=0.0, lam=1.6, n=0.2, a=1.25)
>>> res = solve_pipe_flow(G=0.6e6, R=0.01, fluid=fluid)   # G in Pa/m, R in m
>>> round(res.Q * 1e6, 3)          # flow rate in mL/s
"""

from .model import (
    CarreauYasuda,
    EXAMPLE_FLUID,
    ReducedParameters,
    reduced_parameters,
    wall_shear_rate,
    wall_shear_stress,
)
from .solution import (
    PipeFlowResult,
    VelocityProfile,
    flow_rate_curve,
    pressure_gradient_from_flow_rate,
    solve_pipe_flow,
    velocity_profile,
    eta_tilde,
    r_tilde,
    Phi,
    w,
    u_tilde,
    u_max_tilde,
    u_avg_tilde,
)

__all__ = [
    "CarreauYasuda", "EXAMPLE_FLUID", "ReducedParameters",
    "reduced_parameters", "wall_shear_rate", "wall_shear_stress",
    "PipeFlowResult", "VelocityProfile",
    "solve_pipe_flow", "velocity_profile", "flow_rate_curve",
    "pressure_gradient_from_flow_rate",
    "eta_tilde", "r_tilde", "Phi", "w", "u_tilde", "u_max_tilde", "u_avg_tilde",
]

__version__ = "1.0.0"
