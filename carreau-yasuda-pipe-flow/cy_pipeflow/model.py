"""The Carreau-Yasuda fluid and the wall-shear-rate equation.

Equation numbers refer to

    Y. Wang, "Steady isothermal flow of a Carreau-Yasuda model fluid in a
    straight circular tube", J. Non-Newtonian Fluid Mech. 310 (2022) 104937.
    https://doi.org/10.1016/j.jnnfm.2022.104937

Everything in this module is in physical (dimensional) units.  The
dimensionless solution lives in ``solution.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CarreauYasuda:
    """Five-parameter Carreau-Yasuda viscosity model, Eq. (1).

    Attributes
    ----------
    eta0    : zero-shear-rate viscosity [Pa s]
    eta_inf : infinite-shear-rate viscosity [Pa s]  (may be 0)
    lam     : time constant lambda [s]
    n       : power-law exponent [-]  (n < 1 shear-thinning, n = 1 Newtonian)
    a       : Yasuda transition parameter [-]  (a = 2 gives the Carreau model)
    """

    eta0: float
    eta_inf: float
    lam: float
    n: float
    a: float

    def __post_init__(self):
        if not self.eta0 > 0:
            raise ValueError("eta0 must be positive")
        if self.eta_inf < 0 or self.eta_inf >= self.eta0:
            raise ValueError("eta_inf must satisfy 0 <= eta_inf < eta0")
        if not self.lam > 0:
            raise ValueError("lam must be positive")
        if not self.n > 0:
            raise ValueError("n must be positive")
        if not self.a > 0:
            raise ValueError("a must be positive")

    # -- Eq. (1) -------------------------------------------------------
    def viscosity(self, gamma_dot):
        """Shear viscosity eta(gamma_dot) [Pa s]; gamma_dot in [1/s]."""
        gamma_dot = np.asarray(gamma_dot, dtype=float)
        return self.eta_inf + (self.eta0 - self.eta_inf) * (
            1.0 + (self.lam * gamma_dot) ** self.a
        ) ** ((self.n - 1.0) / self.a)

    def shear_stress(self, gamma_dot):
        """Shear stress eta(gamma_dot) * gamma_dot [Pa], right side of Eq. (3)."""
        return self.viscosity(gamma_dot) * np.asarray(gamma_dot, dtype=float)

    @property
    def power_law_ratio(self) -> float:
        """u_max/u_avg in the power-law (high-G) limit: (3n+1)/(n+1)."""
        return (3.0 * self.n + 1.0) / (self.n + 1.0)


#: The example fluid of the paper (Section 3, taken from Bird et al.).
EXAMPLE_FLUID = CarreauYasuda(eta0=1400.0, eta_inf=0.0, lam=1.60, n=0.2, a=1.25)


def wall_shear_stress(G: float, R: float) -> float:
    """tau_wall = G R / 2 [Pa], Eq. (4).  G in [Pa/m], R in [m]."""
    return 0.5 * G * R


def wall_shear_rate(G: float, R: float, fluid: CarreauYasuda) -> float:
    """Solve the nonlinear Eq. (4), eta(gamma_w) gamma_w = G R / 2, for gamma_w [1/s].

    This is the single numerical step of the 'almost analytical' solution.
    The shear stress is a monotonically increasing function of the shear rate
    for n > 0, so a bracketed root finder is safe.
    """
    if G <= 0 or R <= 0:
        raise ValueError("G and R must be positive")
    tau_w = wall_shear_stress(G, R)

    def residual(g):
        return float(fluid.shear_stress(g)) - tau_w

    # Physics-based bracket: eta_inf <= eta <= eta0 everywhere, hence
    #   tau_w/eta0 <= gamma_w <= tau_w/eta_inf.
    lo = tau_w / fluid.eta0
    if fluid.eta_inf > 0.0:
        hi = tau_w / fluid.eta_inf
    else:
        hi = 10.0 * lo
        while residual(hi) < 0.0:
            hi *= 10.0
    return brentq(residual, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)


@dataclass(frozen=True)
class ReducedParameters:
    """Wall quantities and the reduced (dimensionless) fluid parameters of Table 1.

    gamma_wall : wall shear rate [1/s]
    eta_wall   : viscosity at the wall shear rate [Pa s], Eq. (6)
    eta0_t     : eta0 / eta_wall
    etainf_t   : eta_inf / eta_wall
    lam_t      : lambda * gamma_wall
    """

    gamma_wall: float
    eta_wall: float
    eta0_t: float
    etainf_t: float
    lam_t: float


def reduced_parameters(G: float, R: float, fluid: CarreauYasuda) -> ReducedParameters:
    """Nondimensionalise a given (G, R, fluid) problem, Table 1 and Eq. (6)."""
    gamma_w = wall_shear_rate(G, R, fluid)
    eta_w = float(fluid.viscosity(gamma_w))
    return ReducedParameters(
        gamma_wall=gamma_w,
        eta_wall=eta_w,
        eta0_t=fluid.eta0 / eta_w,
        etainf_t=fluid.eta_inf / eta_w,
        lam_t=fluid.lam * gamma_w,
    )
