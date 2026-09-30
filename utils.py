"""Small reusable utilities for the CFD solver project."""
from __future__ import annotations

import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using partial-pivot Gaussian elimination.

    This is included so the code can reproduce the direct-solution approach
    shown in the instructor's worked finite-difference solutions.  The
    implementation is independently cross-checked against numpy.linalg.solve
    in the test suite.
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D matrix")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must be a 1-D vector with len(b)=A.shape[0]")

    n = A.shape[0]
    aug = np.column_stack((A, b))
    eps = np.finfo(float).eps * max(1.0, np.linalg.norm(A, ord=np.inf))

    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) <= eps:
            raise np.linalg.LinAlgError("Singular or ill-conditioned matrix")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]

        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            if factor != 0.0:
                aug[row, col:] -= factor * aug[col, col:]

    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        rhs = aug[row, -1] - np.dot(aug[row, row + 1:n], x[row + 1:n])
        x[row] = rhs / aug[row, row]
    return x


def max_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Maximum absolute algebraic residual."""
    return float(np.max(np.abs(np.asarray(A) @ np.asarray(x) - np.asarray(b))))


def compare(numerical: np.ndarray, analytical: np.ndarray):
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)
    if numerical.shape != analytical.shape:
        raise ValueError("Arrays must have the same shape")
    abs_error = np.abs(numerical - analytical)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_error = np.where(
            np.abs(analytical) > 1e-14,
            abs_error / np.abs(analytical) * 100.0,
            0.0,
        )
    return abs_error, pct_error


def print_matrix(A: np.ndarray, b: np.ndarray) -> None:
    print("\n[A] =")
    print(np.array2string(np.asarray(A), precision=8, suppress_small=True))
    print("\n[b] =")
    print(np.array2string(np.asarray(b), precision=8, suppress_small=True))


def print_1d_table(x, numerical, analytical=None, variable="T", unit="") -> None:
    x = np.asarray(x)
    numerical = np.asarray(numerical)
    if analytical is None:
        print(f"\nNode    x              {variable}{unit}")
        print("-" * 48)
        for i, (xi, vi) in enumerate(zip(x, numerical), 1):
            print(f"{i:<8}{xi:<15.8f}{vi:.8f}")
        return

    abs_error, pct_error = compare(numerical, analytical)
    print(f"\nNode    x              Numerical       Analytical      Abs. Error      Error %")
    print("-" * 82)
    for i, (xi, vn, va, ea, ep) in enumerate(zip(x, numerical, analytical, abs_error, pct_error), 1):
        print(f"{i:<8}{xi:<15.8f}{vn:<16.8f}{va:<16.8f}{ea:<16.8f}{ep:.6f}")


def plot_1d(x, numerical, analytical=None, title="1D CFD Solution", ylabel="Value", show=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(x, numerical, marker="o", label="Numerical")
    if analytical is not None:
        plt.plot(x, analytical, marker="s", linestyle="--", label="Analytical")
    plt.xlabel("x (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
