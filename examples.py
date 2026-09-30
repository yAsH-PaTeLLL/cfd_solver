"""Course-problem examples and analytical cross-checks."""
from __future__ import annotations

import math
import numpy as np

from fdm import solve_steady_1d_heat, solve_transient_1d_heat_explicit, solve_transient_1d_heat_implicit
from fvm import BC, dirichlet, neumann_flux, convection, solve_steady_diffusion, solve_transient_diffusion, solve_steady_convection_diffusion, solve_transient_convection_diffusion_implicit
from utils import compare, print_matrix, print_1d_table


def mst1_q3_fdm():
    """MST-1 Q3 exactly as printed in the 2026 paper."""
    r = solve_steady_1d_heat(0.5, 5, 1.0, 1273.0, 1.0, 373.0, 1000.0)
    # For question as printed: T''-T=-1273, T(0)=373, T'(L)=1000.
    # T=C1 exp(x)+C2 exp(-x)+1273.
    L = 0.5
    M = np.array([[1.0, 1.0], [math.exp(L), -math.exp(-L)]])
    c = np.linalg.solve(M, np.array([373.0-1273.0, 1000.0]))
    analytical = c[0]*np.exp(r["x"]) + c[1]*np.exp(-r["x"]) + 1273.0
    ea, ep = compare(r["T"], analytical)
    return r, analytical, ea, ep


def mst1_q4_transient():
    """MST-1 Q4: first implicit time level using the instructor's setup."""
    kwargs = dict(length=0.005, nodes=6, k=0.25, rho=1300.0, cp=2000.0, dt=10.40,
                  total_time=10.40, initial_temperature=30.0, right_temperature=250.0)
    implicit = solve_transient_1d_heat_implicit(**kwargs)
    return implicit, None


def mst2_q1_fvm():
    """MST-2 Q1: steady FVM heat conduction, as printed."""
    r = solve_steady_diffusion(
        length=0.2, cells=4, Gamma=1.0, source_A=1000.0, source_B=5.0,
        left=dirichlet(300.0), right=neumann_flux(1000.0)
    )
    return r


def assignment2_q1_fvm_robin():
    """Assignment-2 Q1 using the physically stated slab [-a,a] with 4 CVs.

    The uploaded assignment's analytical expression T=170-5000*x^2 is the
    half-slab form with x measured from the symmetry plane; its table also
    lists x=0..0.2, which is inconsistent with 2a=0.2 (a=0.1).  The solver
    therefore uses the stated geometry x in [-0.1,0.1] for the physical test.
    """
    # Cell-centred solver is naturally written on [0,0.2]. We use a shifted
    # variable centered on 0.1 and map the symmetric boundary conditions.
    # For the test, solve directly with x in [0,0.2]. The exact symmetric
    # analytical solution in this coordinate is 120 + 1000x - 5000x^2.
    r = solve_steady_diffusion(
        length=0.2, cells=4, Gamma=2.0, source_A=20000.0, source_B=0.0,
        left=convection(20.0, 20.0), right=convection(20.0, 20.0)
    )
    xa = r["x"]
    analytical = 120.0 + 1000.0 * xa - 5000.0 * xa**2
    return r, analytical


def mst2_q3_convection_diffusion():
    """MST-2 Q3 central-difference FVM setup."""
    common = dict(length=0.1, cells=5, rho=12.0, u=1.0, Gamma=0.24,
                  left=dirichlet(1.0), right=dirichlet(0.0))
    central = solve_steady_convection_diffusion(**common, scheme="central")
    upwind = solve_steady_convection_diffusion(**common, scheme="upwind")
    Pe = common["rho"] * common["u"] * common["length"] / common["Gamma"]
    exact = 1.0 + (0.0 - 1.0) * (np.exp(Pe * central["x"] / common["length"]) - 1.0) / (np.exp(Pe) - 1.0)
    return central, upwind, exact


def mst2_q4_transient_cd():
    """MST-2 Q4 fully implicit transient FVM setup.

    A right zero-gradient boundary is represented as a zero diffusive flux;
    for positive velocity the outgoing convective flux is handled by the
    upwind/central face scheme. The assignment requests central differencing.
    """
    r = solve_transient_convection_diffusion_implicit(
        length=1.5, cells=3, rho=1.0, u=2.0, Gamma=0.03,
        rho_cp=1.0, dt=0.1, total_time=0.5, initial_value=0.0,
        left_value=0.0, right_bc=neumann_flux(0.0), scheme="central"
    )
    return r
