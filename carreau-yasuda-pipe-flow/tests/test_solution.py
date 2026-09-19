"""Checks on the analytical solution: limits, closure against quadrature, inversion."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cy_pipeflow import (CarreauYasuda, EXAMPLE_FLUID, reduced_parameters, solve_pipe_flow,
                         velocity_profile, pressure_gradient_from_flow_rate, wall_shear_rate,
                         u_tilde)
from cy_pipeflow.numerical import (u_avg_tilde_numerical, u_max_tilde_numerical,
                                   u_tilde_numerical)

R = 0.01
FLUIDS = [
    EXAMPLE_FLUID,
    CarreauYasuda(eta0=1400.0, eta_inf=5.0, lam=1.6, n=0.2, a=1.25),   # eta_inf > 0
    CarreauYasuda(eta0=50.0, eta_inf=0.1, lam=0.5, n=0.4, a=2.0),      # Carreau (a = 2)
    CarreauYasuda(eta0=10.0, eta_inf=0.0, lam=0.3, n=0.6, a=0.4),      # Cross-like (a = 1-n)
]


def test_wall_shear_rate_satisfies_eq4():
    for f in FLUIDS:
        for G in [1e3, 1e5, 1e7]:
            gw = wall_shear_rate(G, R, f)
            assert np.isclose(float(f.shear_stress(gw)), 0.5 * G * R, rtol=1e-11)


@pytest.mark.parametrize("fluid", FLUIDS)
@pytest.mark.parametrize("G", [1e3, 3e5, 3e6])
def test_hypergeometric_matches_quadrature(fluid, G):
    rp = reduced_parameters(G, R, fluid)
    res = solve_pipe_flow(G, R, fluid)
    assert np.isclose(res.u_max_tilde, u_max_tilde_numerical(rp, fluid), rtol=1e-9)
    assert np.isclose(res.u_avg_tilde, u_avg_tilde_numerical(rp, fluid), rtol=1e-9)
    s = np.linspace(0, 1, 25)
    assert np.allclose(u_tilde(s, rp, fluid), u_tilde_numerical(s, rp, fluid), atol=1e-10)


def test_newtonian_limit_at_small_G():
    res = solve_pipe_flow(1.0, R, EXAMPLE_FLUID)          # lambda*gamma_w << 1
    assert np.isclose(res.u_max_tilde, 0.5, rtol=1e-5)
    assert np.isclose(res.u_avg_tilde, 0.25, rtol=1e-5)
    assert np.isclose(res.ratio, 2.0, rtol=1e-5)
    # Hagen-Poiseuille with eta0
    assert np.isclose(res.Q, np.pi * R**4 * 1.0 / (8 * EXAMPLE_FLUID.eta0), rtol=1e-4)


def test_power_law_limit_at_large_G():
    f = EXAMPLE_FLUID
    res = solve_pipe_flow(2e7, R, f)
    assert np.isclose(res.ratio, f.power_law_ratio, rtol=1e-4)


def test_flow_rate_inversion_roundtrip():
    for f in FLUIDS:
        for G in [2e4, 5e5]:
            Q = solve_pipe_flow(G, R, f).Q
            assert np.isclose(pressure_gradient_from_flow_rate(Q, R, f), G, rtol=1e-9)


def test_profile_boundary_values():
    prof = velocity_profile(0.6e6, R, EXAMPLE_FLUID)
    assert prof.r[0] == 0.0 and np.isclose(prof.r[-1], R)
    assert np.isclose(prof.u[-1], 0.0, atol=1e-14)
    assert np.isclose(prof.u[0], prof.result.u_max)
    assert np.all(np.diff(prof.r) > 0) and np.all(np.diff(prof.u) <= 0)
    # Fig. 3 of the paper: reduced centreline velocity about 0.169
    assert abs(prof.result.u_max_tilde - 0.1686) < 5e-4


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        CarreauYasuda(eta0=1.0, eta_inf=2.0, lam=1.0, n=0.5, a=2.0)
    with pytest.raises(ValueError):
        CarreauYasuda(eta0=1.0, eta_inf=0.0, lam=-1.0, n=0.5, a=2.0)
