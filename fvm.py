"""One-dimensional finite-volume solvers for the course project.

Conventions:
- Cell-centered uniform control volumes are used for the FVM core.
- A 1D domain of length L with N control volumes has dx=L/N and cell centers
  at (i+1/2)dx.
- Constant-area problems use A_face=1 unless an area is explicitly supplied.
- Diffusion coefficient is written Gamma (for heat conduction, Gamma=k).
- For convection-diffusion, F=rho*u*A and D=Gamma*A/dx at an interior face.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from utils import gaussian_elimination


@dataclass(frozen=True)
class BC:
    kind: str
    value: float
    h: float | None = None
    ambient: float | None = None


def dirichlet(value: float) -> BC:
    return BC("dirichlet", float(value))


def neumann_flux(value: float) -> BC:
    return BC("flux", float(value))


def convection(h: float, ambient: float) -> BC:
    return BC("convection", 0.0, float(h), float(ambient))


def _validate_bcs(left: BC, right: BC):
    if left.kind not in {"dirichlet", "flux", "convection"}:
        raise ValueError("Unsupported left boundary type")
    if right.kind not in {"dirichlet", "flux", "convection"}:
        raise ValueError("Unsupported right boundary type")
    if left.kind == "convection" and (left.h is None or left.ambient is None or left.h < 0):
        raise ValueError("Invalid left convection boundary")
    if right.kind == "convection" and (right.h is None or right.ambient is None or right.h < 0):
        raise ValueError("Invalid right convection boundary")


def _boundary_diffusion_coeff(Gamma, area, dx):
    # Boundary is half a control-volume width from the adjacent cell center.
    return 2.0 * Gamma * area / dx


def solve_steady_diffusion(
    length: float,
    cells: int,
    Gamma: float,
    source_A: float = 0.0,
    source_B: float = 0.0,
    left: BC = BC("dirichlet", 0.0),
    right: BC = BC("dirichlet", 0.0),
    area: float = 1.0,
    solver: str = "gauss",
):
    """Steady 1D diffusion with a linear source S=A-B*phi.

    Integral equation over each cell:
        (Gamma A phi_x)_e - (Gamma A phi_x)_w + S_V = 0.
    The source is linearized exactly as S_C + S_P*phi with S_P=-B.
    """
    if cells < 1 or length <= 0 or Gamma <= 0 or area <= 0:
        raise ValueError("Invalid FVM inputs")
    _validate_bcs(left, right)
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    Af = area
    Acoef = np.zeros((cells, cells), float)
    b = np.zeros(cells, float)
    D = Gamma * Af / dx
    D_b = _boundary_diffusion_coeff(Gamma, Af, dx)
    V = Af * dx

    for p in range(cells):
        aW = D if p > 0 else 0.0
        aE = D if p < cells - 1 else 0.0
        aP = aW + aE - (-source_B) * V  # aP = sum(a_nb) - S_P*V; S_P=-B
        rhs = source_A * V

        if p == 0:
            if left.kind == "dirichlet":
                aP += D_b
                rhs += D_b * left.value
            elif left.kind == "flux":
                rhs += left.value * Af
            else:  # convection
                H = left.h * Af
                aP += (D_b * H) / (D_b + H)
                rhs += (D_b * H) / (D_b + H) * left.ambient
        else:
            Acoef[p, p - 1] = -aW

        if p == cells - 1:
            if right.kind == "dirichlet":
                aP += D_b
                rhs += D_b * right.value
            elif right.kind == "flux":
                rhs += right.value * Af
            else:
                H = right.h * Af
                aP += (D_b * H) / (D_b + H)
                rhs += (D_b * H) / (D_b + H) * right.ambient
        else:
            Acoef[p, p + 1] = -aE

        Acoef[p, p] = aP
        b[p] = rhs

    phi = gaussian_elimination(Acoef, b) if solver == "gauss" else np.linalg.solve(Acoef, b)
    return {"x": x, "phi": phi, "A": Acoef, "b": b, "dx": dx}


def solve_transient_diffusion(
    length: float,
    cells: int,
    Gamma: float,
    rho_cp: float,
    dt: float,
    total_time: float,
    initial_value: float,
    source_A: float = 0.0,
    source_B: float = 0.0,
    left: BC = BC("dirichlet", 0.0),
    right: BC = BC("dirichlet", 0.0),
    area: float = 1.0,
    scheme: str = "implicit",
    stability_limit: float = 0.5,
):
    """Transient 1D diffusion, cell-centred FVM.

    Explicit uses forward Euler. Implicit uses backward Euler.  The same
    linear source and boundary treatments are used in both.
    """
    if rho_cp <= 0 or dt <= 0 or total_time <= 0:
        raise ValueError("rho_cp, dt and total_time must be positive")
    _validate_bcs(left, right)
    if scheme not in {"explicit", "implicit"}:
        raise ValueError("scheme must be explicit or implicit")
    if not np.isclose(round(total_time / dt) * dt, total_time, rtol=1e-10, atol=1e-12):
        raise ValueError("total_time must be an integer multiple of dt")
    steps = int(round(total_time / dt))
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    alpha = Gamma / rho_cp
    Fo = alpha * dt / dx**2
    D = Gamma * area / dx
    D_b = _boundary_diffusion_coeff(Gamma, area, dx)
    V = area * dx
    storage = rho_cp * V / dt

    def build_matrix_implicit(storage=storage):
        M = np.zeros((cells, cells), float)
        for p in range(cells):
            aW = D if p > 0 else 0.0
            aE = D if p < cells - 1 else 0.0
            aP = storage + aW + aE - (-source_B) * V
            if p == 0:
                if left.kind == "dirichlet":
                    aP += D_b
                elif left.kind == "convection":
                    aP += (D_b * left.h * area) / (D_b + left.h * area)
                # flux affects RHS only
            if p == cells - 1:
                if right.kind == "dirichlet":
                    aP += D_b
                elif right.kind == "convection":
                    aP += (D_b * right.h * area) / (D_b + right.h * area)
            M[p, p] = aP
            if p > 0:
                M[p, p - 1] = -aW
            if p < cells - 1:
                M[p, p + 1] = -aE
        return M

    if scheme == "explicit":
        # Positive-coefficient rule: the coefficient of the old value phi_P^0,
        #   rho*c*V/dt - (sum of neighbour/boundary conductances + B*V),
        # must not be negative in ANY cell.  Cells touching a fixed-value or
        # convective boundary carry an extra half-cell conductance, so the limit
        # there is tighter than the interior Fo <= 0.5 (Fo <= 1/3 for a Dirichlet end).
        a_sum = np.diag(build_matrix_implicit(storage=0.0))
        with np.errstate(divide="ignore"):
            dt_max = float(np.min(np.where(a_sum > 0, rho_cp * V / a_sum, np.inf)))
        if dt > dt_max * (1 + 1e-12) or Fo > stability_limit + 1e-12:
            raise ValueError(
                f"Explicit FVM unstable/oscillatory: dt={dt:.6g} > dt_max={dt_max:.6g} "
                f"(Fo={Fo:.6g}, largest allowed Fo={dt_max*alpha/dx**2:.6g})"
            )

    phi = np.full(cells, float(initial_value))
    history = [phi.copy()]
    M = build_matrix_implicit() if scheme == "implicit" else None

    for _ in range(steps):
        old = phi.copy()
        if scheme == "explicit":
            new = old.copy()
            for p in range(cells):
                net = source_A * V - source_B * old[p] * V
                # west flux contribution in the integrated equation
                if p == 0:
                    if left.kind == "dirichlet":
                        net += D_b * (left.value - old[p])
                    elif left.kind == "flux":
                        net += left.value * area
                    else:
                        H = left.h * area
                        keq = D_b * H / (D_b + H)
                        net += keq * (left.ambient - old[p])
                else:
                    net += D * (old[p - 1] - old[p])
                if p == cells - 1:
                    if right.kind == "dirichlet":
                        net += D_b * (right.value - old[p])
                    elif right.kind == "flux":
                        net += right.value * area
                    else:
                        H = right.h * area
                        keq = D_b * H / (D_b + H)
                        net += keq * (right.ambient - old[p])
                else:
                    net += D * (old[p + 1] - old[p])
                new[p] = old[p] + dt * net / (rho_cp * V)
            phi = new
        else:
            rhs = storage * old + source_A * V
            if left.kind == "dirichlet":
                rhs[0] += D_b * left.value
            elif left.kind == "flux":
                rhs[0] += left.value * area
            elif left.kind == "convection":
                keq = D_b * left.h * area / (D_b + left.h * area)
                rhs[0] += keq * left.ambient
            if right.kind == "dirichlet":
                rhs[-1] += D_b * right.value
            elif right.kind == "flux":
                rhs[-1] += right.value * area
            elif right.kind == "convection":
                keq = D_b * right.h * area / (D_b + right.h * area)
                rhs[-1] += keq * right.ambient
            phi = gaussian_elimination(M, rhs)
        history.append(phi.copy())

    return {"x": x, "times": np.arange(steps + 1) * dt, "history": np.asarray(history), "dx": dx, "Fo": Fo, "matrix": M}


def _convection_coefficients(rho, u, Gamma, area, dx, scheme):
    F = rho * u * area
    D = Gamma * area / dx
    if scheme == "central":
        aW = D + F / 2.0
        aE = D - F / 2.0
    elif scheme == "upwind":
        aW = D + max(F, 0.0)
        aE = D + max(-F, 0.0)
    else:
        raise ValueError("scheme must be central or upwind")
    return F, D, aW, aE


def _dirichlet_face_extra(F, scheme):
    """Extra convective diagonal term for a boundary cell whose boundary face has a fixed value.

    Returns (west_extra, east_extra).  Derived from the face flux balance:
      central : face value = boundary value on both ends  ->  (+F, -F)   for any sign of F
      upwind  : boundary value is used only where flow ENTERS the domain -> (max(F,0), max(-F,0))
    """
    if scheme == "central":
        return F, -F
    return max(F, 0.0), max(-F, 0.0)


def solve_steady_convection_diffusion(
    length: float,
    cells: int,
    rho: float,
    u: float,
    Gamma: float,
    left: BC,
    right: BC,
    scheme: str = "central",
    source_A: float = 0.0,
    source_B: float = 0.0,
    area: float = 1.0,
):
    """Steady 1D convection-diffusion, constant rho/u/Gamma, FVM."""
    if cells < 1 or length <= 0 or rho <= 0 or Gamma <= 0 or area <= 0:
        raise ValueError("Invalid inputs")
    _validate_bcs(left, right)
    if left.kind != "dirichlet" or right.kind != "dirichlet":
        raise NotImplementedError("Current convection-diffusion example uses Dirichlet endpoints")
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    F, D, aW, aE = _convection_coefficients(rho, u, Gamma, area, dx, scheme)
    V = area * dx
    Ab = np.zeros((cells, cells), float)
    b = np.zeros(cells, float)

    for p in range(cells):
        rhs = source_A * V
        if p == 0 and p == cells - 1:
            raise ValueError("Use at least 2 control volumes for convection-diffusion")

        if p == 0:
            # West boundary: replace the west neighbour by the boundary face.
            Db = 2.0 * Gamma * area / dx
            aE_local = aE
            wx, _ = _dirichlet_face_extra(F, scheme)
            AP = aE_local + Db + wx
            rhs += (Db + wx) * left.value
            Ab[p, p + 1] = -aE_local
        elif p == cells - 1:
            # East boundary: replace the east neighbour by the boundary face.
            Db = 2.0 * Gamma * area / dx
            aW_local = aW
            _, ex = _dirichlet_face_extra(F, scheme)
            AP = aW_local + Db + ex
            rhs += (Db + ex) * right.value
            Ab[p, p - 1] = -aW_local
        else:
            AP = aW + aE
            Ab[p, p - 1] = -aW
            Ab[p, p + 1] = -aE

        AP += source_B * V
        Ab[p, p] = AP
        b[p] = rhs

    phi = gaussian_elimination(Ab, b)
    return {"x": x, "phi": phi, "A": Ab, "b": b, "dx": dx, "F": F, "D": D, "cell_Pe": F / D if D else np.inf, "scheme": scheme}


def solve_transient_convection_diffusion_implicit(
    length: float,
    cells: int,
    rho: float,
    u: float,
    Gamma: float,
    rho_cp: float,
    dt: float,
    total_time: float,
    initial_value: float,
    left_value: float,
    right_bc: BC,
    scheme: str = "central",
    source_A: float = 0.0,
    source_B: float = 0.0,
    area: float = 1.0,
):
    """Fully implicit transient 1D convection-diffusion FVM."""
    if right_bc.kind not in {"dirichlet", "flux", "convection"}:
        raise ValueError("Unsupported right boundary")
    if not np.isclose(round(total_time / dt) * dt, total_time, rtol=1e-10, atol=1e-12):
        raise ValueError("total_time must be an integer multiple of dt")
    steps = int(round(total_time / dt))
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    V = area * dx
    F, D, aW, aE = _convection_coefficients(rho, u, Gamma, area, dx, scheme)
    storage = rho_cp * V / dt
    M = np.zeros((cells, cells), float)

    Db = 2.0 * Gamma * area / dx
    wx, ex = _dirichlet_face_extra(F, scheme)
    for p in range(cells):
        if p == 0 and p == cells - 1:
            raise ValueError("Use at least 2 control volumes for transient convection-diffusion")
        if p == 0:
            AP = storage + aE + Db + wx
            M[p, p + 1] = -aE
        elif p == cells - 1:
            if right_bc.kind == "dirichlet":
                AP = storage + aW + Db + ex
            elif right_bc.kind == "flux":
                # Outflow boundary: face value taken equal to the cell value (zero-gradient
                # extrapolation), prescribed diffusive flux goes to the RHS.
                AP = storage + aW
            else:
                keq = Db * right_bc.h * area / (Db + right_bc.h * area)
                AP = storage + aW + keq
            M[p, p - 1] = -aW
        else:
            AP = storage + aW + aE
            M[p, p - 1] = -aW
            M[p, p + 1] = -aE
        AP += source_B * V
        M[p, p] = AP

    phi = np.full(cells, float(initial_value))
    history = [phi.copy()]

    for _ in range(steps):
        rhs = storage * phi + source_A * V
        rhs[0] += (Db + wx) * left_value
        if right_bc.kind == "dirichlet":
            rhs[-1] += (Db + ex) * right_bc.value
        elif right_bc.kind == "flux":
            rhs[-1] += right_bc.value * area
        else:
            keq = Db * right_bc.h * area / (Db + right_bc.h * area)
            rhs[-1] += keq * right_bc.ambient
        phi = gaussian_elimination(M, rhs)
        history.append(phi.copy())

    return {"x": x, "times": np.arange(steps + 1) * dt, "history": np.asarray(history), "matrix": M, "dx": dx, "F": F, "D": D, "cell_Pe": F/D if D else np.inf, "scheme": scheme}
