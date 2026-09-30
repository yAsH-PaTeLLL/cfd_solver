"""One-dimensional finite-difference solvers used in the CFD course project."""
from __future__ import annotations

import numpy as np
from utils import gaussian_elimination


def solve_steady_1d_heat(
    length: float,
    nodes: int,
    k: float,
    source_A: float,
    source_B: float,
    left_temperature: float,
    right_heat_flux: float,
    solver: str = "gauss",
):
    """Solve k*T'' + (A-B*T)=0 with T(0)=TL and k*T'(L)=qL.

    The spatial discretization and first-order right Neumann treatment match
    the 1D steady FDM workflow used in the uploaded MST-1 solution.
    """
    if nodes < 3 or length <= 0 or k <= 0:
        raise ValueError("Require nodes>=3, length>0 and k>0")
    dx = length / (nodes - 1)
    x = np.linspace(0.0, length, nodes)
    A = np.zeros((nodes, nodes), float)
    b = np.zeros(nodes, float)

    A[0, 0] = 1.0
    b[0] = left_temperature

    diag = 2.0 + source_B * dx * dx / k
    rhs_source = source_A * dx * dx / k
    for i in range(1, nodes - 1):
        A[i, i - 1] = -1.0
        A[i, i] = diag
        A[i, i + 1] = -1.0
        b[i] = rhs_source

    A[-1, -2] = -1.0
    A[-1, -1] = 1.0
    b[-1] = right_heat_flux * dx / k

    T = gaussian_elimination(A, b) if solver == "gauss" else np.linalg.solve(A, b)
    q_left_forward = -k * (T[1] - T[0]) / dx
    return {"x": x, "T": T, "A": A, "b": b, "dx": dx, "q_left": q_left_forward}


def _validate_time_inputs(length, nodes, k, rho, cp, dt, total_time):
    if nodes < 3 or length <= 0 or k <= 0 or rho <= 0 or cp <= 0 or dt <= 0 or total_time <= 0:
        raise ValueError("Invalid physical/grid/time inputs")
    steps_float = total_time / dt
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps * dt, total_time, rtol=1e-10, atol=1e-12):
        raise ValueError("total_time must be an integer multiple of dt")
    dx = length / (nodes - 1)
    alpha = k / (rho * cp)
    fo = alpha * dt / dx**2
    return dx, alpha, fo, steps


def _bc(kind: str, value: float | None = None):
    kind = kind.lower()
    if kind not in {"dirichlet", "neumann"}:
        raise ValueError("Boundary kind must be 'dirichlet' or 'neumann'")
    return {"kind": kind, "value": None if value is None else float(value)}


def _validate_transient_bcs(left_bc, right_bc):
    for name, bc in (("left", left_bc), ("right", right_bc)):
        if not isinstance(bc, dict) or bc.get("kind") not in {"dirichlet", "neumann"}:
            raise ValueError(f"{name}_bc must be a dict made with fdm._bc-style data")
        if bc.get("kind") == "dirichlet" and bc.get("value") is None:
            raise ValueError(f"{name} Dirichlet boundary requires a value")
        if bc.get("kind") == "neumann" and bc.get("value") is None:
            raise ValueError(f"{name} Neumann boundary requires a derivative dT/dx value")


def solve_transient_1d_heat_explicit_general(
    length: float,
    nodes: int,
    k: float,
    rho: float,
    cp: float,
    dt: float,
    total_time: float,
    initial_temperature: float,
    left_bc: dict,
    right_bc: dict,
    stability_limit: float = 0.5,
):
    """Explicit FTCS 1D heat equation with Dirichlet/Neumann end conditions.

    Neumann conditions are imposed with a second-order ghost-point relation:
        dT/dx = g.
    For g=0 this reduces to the symmetry/adiabatic condition used in MST-1.
    """
    dx, alpha, fo, steps = _validate_time_inputs(length, nodes, k, rho, cp, dt, total_time)
    _validate_transient_bcs(left_bc, right_bc)
    if fo > stability_limit + 1e-12:
        raise ValueError(f"Explicit FTCS unstable: Fo={fo:.8f} > {stability_limit}")

    x = np.linspace(0.0, length, nodes)
    T = np.full(nodes, float(initial_temperature))
    history = [T.copy()]

    def impose_bc(arr):
        if left_bc["kind"] == "dirichlet":
            arr[0] = left_bc["value"]
        else:
            g = left_bc["value"]
            # Centered first derivative at boundary: (T1-T_-1)/(2dx)=g,
            # hence ghost T_-1 = T1 - 2g dx, and T'' at boundary becomes
            # 2(T1-T0)/dx^2 - 2g/dx.
            arr[0] = arr[1] - g * dx
        if right_bc["kind"] == "dirichlet":
            arr[-1] = right_bc["value"]
        else:
            g = right_bc["value"]
            arr[-1] = arr[-2] + g * dx

    impose_bc(T)
    history[0] = T.copy()

    for _ in range(steps):
        old = T.copy()
        new = old.copy()

        # Interior nodes.
        new[1:-1] = old[1:-1] + fo * (old[2:] - 2.0 * old[1:-1] + old[:-2])

        if left_bc["kind"] == "dirichlet":
            new[0] = left_bc["value"]
        else:
            g = left_bc["value"]
            new[0] = old[0] + 2.0 * fo * (old[1] - old[0] - g * dx)

        if right_bc["kind"] == "dirichlet":
            new[-1] = right_bc["value"]
        else:
            g = right_bc["value"]
            new[-1] = old[-1] + 2.0 * fo * (old[-2] - old[-1] + g * dx)

        T = new
        history.append(T.copy())

    times = np.arange(steps + 1, dtype=float) * dt
    return {"x": x, "times": times, "history": np.asarray(history), "alpha": alpha, "Fo": fo, "dx": dx, "dt": dt}


def solve_transient_1d_heat_implicit_general(
    length: float,
    nodes: int,
    k: float,
    rho: float,
    cp: float,
    dt: float,
    total_time: float,
    initial_temperature: float,
    left_bc: dict,
    right_bc: dict,
    solver: str = "gauss",
):
    """Fully implicit 1D heat equation with Dirichlet/Neumann end conditions."""
    dx, alpha, fo, steps = _validate_time_inputs(length, nodes, k, rho, cp, dt, total_time)
    _validate_transient_bcs(left_bc, right_bc)

    x = np.linspace(0.0, length, nodes)
    M = np.zeros((nodes, nodes), float)

    # Left boundary equation.
    if left_bc["kind"] == "dirichlet":
        M[0, 0] = 1.0
    else:
        M[0, 0] = 1.0
        M[0, 1] = -1.0

    # Interior nodes.
    for i in range(1, nodes - 1):
        M[i, i - 1] = -fo
        M[i, i] = 1.0 + 2.0 * fo
        M[i, i + 1] = -fo

    # Right boundary equation.
    if right_bc["kind"] == "dirichlet":
        M[-1, -1] = 1.0
    else:
        M[-1, -2] = -1.0
        M[-1, -1] = 1.0

    T = np.full(nodes, float(initial_temperature))
    history = [T.copy()]

    # Apply boundary values to initial state, as would be expected for a physical
    # problem where the boundary conditions are already active at t=0.
    if left_bc["kind"] == "dirichlet":
        T[0] = left_bc["value"]
    else:
        T[0] = T[1] - left_bc["value"] * dx
    if right_bc["kind"] == "dirichlet":
        T[-1] = right_bc["value"]
    else:
        T[-1] = T[-2] + right_bc["value"] * dx
    history[0] = T.copy()

    for _ in range(steps):
        rhs = np.empty(nodes, float)

        if left_bc["kind"] == "dirichlet":
            rhs[0] = left_bc["value"]
        else:
            rhs[0] = -left_bc["value"] * dx   # T0 - T1 = -g*dx  (dT/dx = g at x=0)

        rhs[1:-1] = T[1:-1]

        if right_bc["kind"] == "dirichlet":
            rhs[-1] = right_bc["value"]
        else:
            rhs[-1] = right_bc["value"] * dx

        T = gaussian_elimination(M, rhs) if solver == "gauss" else np.linalg.solve(M, rhs)
        history.append(T.copy())

    times = np.arange(steps + 1, dtype=float) * dt
    return {"x": x, "times": times, "history": np.asarray(history), "alpha": alpha, "Fo": fo, "dx": dx, "dt": dt, "matrix": M}


# Backward-compatible wrappers used by earlier examples/tests.
def solve_transient_1d_heat_explicit(
    length, nodes, k, rho, cp, dt, total_time, initial_temperature,
    left_bc="adiabatic", right_temperature=250.0, stability_limit=0.5,
):
    if left_bc != "adiabatic":
        raise NotImplementedError("Use solve_transient_1d_heat_explicit_general for arbitrary BCs")
    return solve_transient_1d_heat_explicit_general(
        length, nodes, k, rho, cp, dt, total_time, initial_temperature,
        {"kind": "neumann", "value": 0.0},
        {"kind": "dirichlet", "value": right_temperature},
        stability_limit=stability_limit,
    )


def solve_transient_1d_heat_implicit(
    length, nodes, k, rho, cp, dt, total_time, initial_temperature,
    left_bc="adiabatic", right_temperature=250.0, solver="gauss",
):
    if left_bc != "adiabatic":
        raise NotImplementedError("Use solve_transient_1d_heat_implicit_general for arbitrary BCs")
    return solve_transient_1d_heat_implicit_general(
        length, nodes, k, rho, cp, dt, total_time, initial_temperature,
        {"kind": "neumann", "value": 0.0},
        {"kind": "dirichlet", "value": right_temperature},
        solver=solver,
    )
