"""Analytical solution of the Carreau-Yasuda Hagen-Poiseuille problem.

Dimensionless part: Eqs. (8)-(31) of Wang, JNNFM 310 (2022) 104937,
expressed with the Gauss hypergeometric function 2F1 (scipy.special.hyp2f1).

Dimensional part: thin wrappers that convert back to SI units with the
scales of Table 1 (velocity scale R*gamma_wall, flow rate pi R^3 gamma_wall).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1

from .model import CarreauYasuda, ReducedParameters, reduced_parameters, wall_shear_stress


# ----------------------------------------------------------------------
# Dimensionless solution (s = gamma_dot / gamma_wall in [0, 1])
# ----------------------------------------------------------------------

def eta_tilde(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """Reduced viscosity eta/eta_wall as a function of s, Eq. (9)."""
    s = np.asarray(s, dtype=float)
    return rp.etainf_t + (rp.eta0_t - rp.etainf_t) * (
        1.0 + (rp.lam_t * s) ** fluid.a
    ) ** ((fluid.n - 1.0) / fluid.a)


def r_tilde(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """Reduced radial position r/R as a function of s, Eq. (8):  r_t = s*eta_t(s)."""
    return np.asarray(s, dtype=float) * eta_tilde(s, rp, fluid)


def Phi(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """Phi(s) = s^2 2F1[2/a, (1-n)/a; (a+2)/a; -(lam_t s)^a], Eq. (19)."""
    s = np.asarray(s, dtype=float)
    a, n = fluid.a, fluid.n
    return s**2 * hyp2f1(2.0 / a, (1.0 - n) / a, (a + 2.0) / a, -(rp.lam_t * s) ** a)


def w(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """w(s) = 1/2 (eta0_t - etainf_t)[Phi(1) - Phi(s)] + 1/2 etainf_t (1 - s^2), Eq. (18)."""
    s = np.asarray(s, dtype=float)
    return (0.5 * (rp.eta0_t - rp.etainf_t) * (Phi(1.0, rp, fluid) - Phi(s, rp, fluid))
            + 0.5 * rp.etainf_t * (1.0 - s**2))


def u_tilde(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """Reduced velocity u/(R gamma_wall) from the parametric solution, Eq. (22)."""
    s = np.asarray(s, dtype=float)
    return 1.0 - s * r_tilde(s, rp, fluid) - w(s, rp, fluid)


def u_max_tilde(rp: ReducedParameters, fluid: CarreauYasuda) -> float:
    """Reduced maximum (centreline) velocity, Eq. (23)."""
    return float(1.0 - 0.5 * (rp.eta0_t - rp.etainf_t) * Phi(1.0, rp, fluid)
                 - 0.5 * rp.etainf_t)


def u_avg_tilde(rp: ReducedParameters, fluid: CarreauYasuda) -> float:
    """Reduced average velocity, Eq. (31), with I1, I2, I3 of Eqs. (28)-(30)."""
    a, n = fluid.a, fluid.n
    d = rp.eta0_t - rp.etainf_t
    z = -rp.lam_t ** a
    I1 = hyp2f1(4.0 / a, 3.0 * (1.0 - n) / a, (a + 4.0) / a, z) * d**3
    I2 = hyp2f1(4.0 / a, 2.0 * (1.0 - n) / a, (a + 4.0) / a, z) * d**2 * rp.etainf_t
    I3 = hyp2f1(4.0 / a, 1.0 * (1.0 - n) / a, (a + 4.0) / a, z) * d * rp.etainf_t**2
    return float((1.0 / 3.0) * (1.0 - 0.25 * (I1 + 3.0 * I2 + 3.0 * I3 + rp.etainf_t**3)))


# ----------------------------------------------------------------------
# Dimensional results
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PipeFlowResult:
    """Integral quantities of the flow for one (G, R, fluid)."""

    G: float            # pressure drop per unit length [Pa/m]
    R: float            # tube radius [m]
    gamma_wall: float   # wall shear rate [1/s]
    eta_wall: float     # viscosity at the wall [Pa s]
    tau_wall: float     # wall shear stress [Pa]
    u_max: float        # centreline velocity [m/s]
    u_avg: float        # average velocity [m/s]
    Q: float            # volume flow rate [m^3/s]
    u_max_tilde: float  # reduced centreline velocity (0.5 in the Newtonian limit)
    u_avg_tilde: float  # reduced average velocity (0.25 in the Newtonian limit)

    @property
    def ratio(self) -> float:
        """u_max / u_avg: 2 for Newtonian, (3n+1)/(n+1) in the power-law limit."""
        return self.u_max / self.u_avg


def solve_pipe_flow(G: float, R: float, fluid: CarreauYasuda) -> PipeFlowResult:
    """Wall quantities, u_max, u_avg and Q for a given pressure gradient.

    Steps: solve Eq. (4) -> Table 1 scalings -> Eqs. (23) and (31).
    """
    return _result_from_reduced(G, R, fluid, reduced_parameters(G, R, fluid))


def _result_from_reduced(G: float, R: float, fluid: CarreauYasuda,
                         rp: ReducedParameters) -> PipeFlowResult:
    """Dimensional integral quantities from an already-solved set of reduced parameters."""
    umt = u_max_tilde(rp, fluid)
    uat = u_avg_tilde(rp, fluid)
    scale = R * rp.gamma_wall                      # velocity scale [m/s]
    return PipeFlowResult(
        G=G, R=R,
        gamma_wall=rp.gamma_wall, eta_wall=rp.eta_wall,
        tau_wall=wall_shear_stress(G, R),
        u_max=scale * umt, u_avg=scale * uat,
        Q=np.pi * R**3 * rp.gamma_wall * uat,      # Eq. (24)
        u_max_tilde=umt, u_avg_tilde=uat,
    )


@dataclass(frozen=True)
class VelocityProfile:
    """Radial distributions for one (G, R, fluid), all arrays over s in [0, 1].

    s        : reduced shear rate (the parameter of Eq. (22))
    r        : radial position [m]
    u        : axial velocity [m/s]
    gamma    : local shear rate [1/s]
    eta      : local viscosity [Pa s]
    tau      : local shear stress [Pa]
    r_tilde  : r / R
    u_tilde  : u / (R gamma_wall)
    """

    s: np.ndarray
    r: np.ndarray
    u: np.ndarray
    gamma: np.ndarray
    eta: np.ndarray
    tau: np.ndarray
    r_tilde: np.ndarray
    u_tilde: np.ndarray
    result: PipeFlowResult


def velocity_profile(G: float, R: float, fluid: CarreauYasuda,
                     n_points: int = 400) -> VelocityProfile:
    """Parametric velocity distribution, Eq. (22), in dimensional form.

    The profile is returned as a function of the parameter s; ``r`` and ``u``
    are therefore not on a uniform radial grid (they are dense near the wall,
    where the velocity changes fastest -- exactly where you want the points).
    Use ``numpy.interp(r_query, profile.r, profile.u)`` for a uniform grid.
    """
    rp = reduced_parameters(G, R, fluid)
    res = _result_from_reduced(G, R, fluid, rp)
    s = np.linspace(0.0, 1.0, n_points)
    rt = r_tilde(s, rp, fluid)
    ut = u_tilde(s, rp, fluid)
    gamma = s * rp.gamma_wall
    return VelocityProfile(
        s=s, r=R * rt, u=R * rp.gamma_wall * ut,
        gamma=gamma, eta=fluid.viscosity(gamma), tau=res.tau_wall * rt,
        r_tilde=rt, u_tilde=ut, result=res,
    )


def flow_rate_curve(G_values, R: float, fluid: CarreauYasuda) -> dict:
    """Q, u_avg, u_max, gamma_wall, ... over an array of pressure gradients.

    Returns a dict of equal-length numpy arrays keyed by the field names of
    ``PipeFlowResult`` plus ``ratio``.
    """
    results = [solve_pipe_flow(float(G), R, fluid) for G in np.asarray(G_values, dtype=float)]
    fields = ["G", "R", "gamma_wall", "eta_wall", "tau_wall",
              "u_max", "u_avg", "Q", "u_max_tilde", "u_avg_tilde"]
    out = {f: np.array([getattr(r, f) for r in results]) for f in fields}
    out["ratio"] = out["u_max"] / out["u_avg"]
    return out


def pressure_gradient_from_flow_rate(Q: float, R: float, fluid: CarreauYasuda) -> float:
    """Inverse problem: the pressure gradient G [Pa/m] that delivers a flow rate Q [m^3/s].

    Useful for capillary rheometry.  Since eta_inf <= eta <= eta0, the answer is
    bracketed by the two Newtonian Hagen-Poiseuille estimates.
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    G_hi = 8.0 * fluid.eta0 * Q / (np.pi * R**4)          # Newtonian with eta0
    G_lo = 8.0 * fluid.eta_inf * Q / (np.pi * R**4)       # Newtonian with eta_inf
    G_lo = max(G_lo, 1e-12 * G_hi)
    return brentq(lambda G: solve_pipe_flow(G, R, fluid).Q - Q, G_lo, G_hi,
                  xtol=1e-14 * G_hi, rtol=1e-13)
