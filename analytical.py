"""Closed-form solutions given in the assignments, used for validation.

Coordinates follow the assignment statements:
  * a1_q3_exact : x measured from the slab CENTRE, -b <= x <= b
  * a2_q1_exact : x measured from the slab CENTRE, -a <= x <= a
  * a2_q3_exact : x measured from the slab CENTRE, -a <= x <= a
The solvers report x from the LEFT face, so callers shift by the half-width.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def a1_q2_exact(x):
    """Assignment-1 Q2: T(x) = 232 e^{sqrt5 x} - 132 e^{-sqrt5 x} + 200 ;  q(0) = -814 W/m^2."""
    x = np.asarray(x, dtype=float)
    return 232 * np.exp(np.sqrt(5) * x) - 132 * np.exp(-np.sqrt(5) * x) + 200


A1_Q2_Q0 = -814.0


def a1_q3_exact(x, t, b=0.005, T0=30, Tb=250, k=.25, rho=1300, cp=2000, n=500):
    """Assignment-1 Q3 series solution (x from slab centre)."""
    alpha = k / (rho * cp)
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x)
    for j in range(1, n + 1):
        lam = (2 * j - 1) * np.pi / (2 * b)
        s += np.sin(lam * b) / (lam * b) * np.cos(lam * x) * np.exp(-alpha * lam ** 2 * t)
    return Tb + (T0 - Tb) * 2 * s


def a2_q1_exact(x_centre):
    """Assignment-2 Q1: T = 170 - 5000 x^2 with x measured from the slab centre."""
    return 170.0 - 5000.0 * np.asarray(x_centre, dtype=float) ** 2


def a2_q2_fin_exact(x, L=.02, h=15, P=.404, k=45, A=.0004, Tw=225, Tinf=25):
    """Insulated-tip fin.  The PDF typesets cos/cos but its heat-loss formula uses tanh,
    i.e. the standard cosh/cosh solution, which is used here."""
    m = np.sqrt(h * P / (k * A))
    return Tinf + (Tw - Tinf) * np.cosh(m * (L - np.asarray(x, dtype=float))) / np.cosh(m * L)


def a2_q2_heat_exact(L=.02, h=15, P=.404, k=45, A=.0004, Tw=225, Tinf=25):
    m = np.sqrt(h * P / (k * A))
    return float(np.sqrt(h * P * k * A) * (Tw - Tinf) * np.tanh(m * L))


def a2_q3_exact(x, t, a=.1, h=20, k=2, rho=1000, cp=20, Ti=120, Tinf=20, n=100):
    """Assignment-2 Q3 series solution (x from slab centre)."""
    alpha = k / (rho * cp)
    Bi = h * a / k
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x)
    for j in range(1, n + 1):
        lo = (j - 1) * np.pi + 1e-10
        hi = (j - .5) * np.pi - 1e-10
        lam = brentq(lambda z: z * np.tan(z) - Bi, lo, hi, maxiter=100)
        coeff = 4 * np.sin(lam) / (2 * lam + np.sin(2 * lam))
        s += coeff * np.cos(lam * x / a) * np.exp(-alpha * lam ** 2 * t / a ** 2)
    return Tinf + (Ti - Tinf) * s
