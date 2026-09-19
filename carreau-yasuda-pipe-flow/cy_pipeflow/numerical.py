"""Independent numerical solutions, used only to validate the hypergeometric formulas.

These integrate Eqs. (17) and (26) directly with adaptive quadrature and
never touch ``hyp2f1``.  They are what the *points* in Fig. 2 and the dashed
line in Fig. 3(b) of the paper stand for.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .model import CarreauYasuda, ReducedParameters
from .solution import eta_tilde, r_tilde


def w_numerical(s: float, rp: ReducedParameters, fluid: CarreauYasuda) -> float:
    """w(s) = integral_s^1 xi eta_t(xi) dxi, Eq. (17), by quadrature."""
    val, _ = quad(lambda xi: xi * float(eta_tilde(xi, rp, fluid)), s, 1.0,
                  epsabs=1e-13, epsrel=1e-13, limit=200)
    return val


def u_tilde_numerical(s, rp: ReducedParameters, fluid: CarreauYasuda):
    """Reduced velocity using the numerical w(s); vectorised over s."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    wn = np.array([w_numerical(si, rp, fluid) for si in s])
    return 1.0 - s * r_tilde(s, rp, fluid) - wn


def u_max_tilde_numerical(rp: ReducedParameters, fluid: CarreauYasuda) -> float:
    """u_max_t = 1 - w(0) by quadrature."""
    return 1.0 - w_numerical(0.0, rp, fluid)


def u_avg_tilde_numerical(rp: ReducedParameters, fluid: CarreauYasuda) -> float:
    """u_avg_t = (1/3)[1 - integral_0^1 r_t(s)^3 ds], Eq. (26), by quadrature."""
    val, _ = quad(lambda s: float(r_tilde(s, rp, fluid)) ** 3, 0.0, 1.0,
                  epsabs=1e-13, epsrel=1e-13, limit=200)
    return (1.0 / 3.0) * (1.0 - val)
