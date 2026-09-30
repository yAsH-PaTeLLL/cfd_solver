"""CFD Classroom Solver - course-aligned Streamlit interface.

Scope is deliberately limited to the problem families required by the uploaded
assignments and MST-1/MST-2 material. It is not a general-purpose CFD package.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fdm import (
    solve_steady_1d_heat,
    solve_transient_1d_heat_explicit_general,
    solve_transient_1d_heat_implicit_general,
)
from fvm import (
    dirichlet,
    neumann_flux,
    convection,
    solve_steady_diffusion,
    solve_transient_diffusion,
    solve_steady_convection_diffusion,
    solve_transient_convection_diffusion_implicit,
)
from utils import max_residual
from analytical import (
    a1_q2_exact, A1_Q2_Q0, a1_q3_exact, a2_q1_exact, a2_q2_fin_exact, a2_q2_heat_exact, a2_q3_exact,
)


st.set_page_config(page_title="CFD Classroom Solver", page_icon="🧮", layout="wide")


def df_1d(x, values, variable="Value", analytical=None):
    data = {
        "Node": np.arange(1, len(x) + 1),
        "x (m)": np.asarray(x, dtype=float),
        variable: np.asarray(values, dtype=float),
    }
    if analytical is not None:
        data["Analytical"] = np.asarray(analytical, dtype=float)
        data["Abs. Error"] = np.abs(data[variable] - data["Analytical"])
    return pd.DataFrame(data)


def download_csv(df, filename):
    return st.download_button(
        "Download table (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_matrix(A, b):
    with st.expander("Show algebraic matrix system"):
        st.code("A · x = b\n\nA =\n" + np.array2string(A, precision=8, suppress_small=True)
                + "\n\nb =\n" + np.array2string(b, precision=8, suppress_small=True))


def render_plot(x, numerical, analytical=None, ylabel="Value", title="Solution"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, numerical, marker="o", label="Numerical")
    if analytical is not None:
        ax.plot(x, analytical, marker="s", linestyle="--", label="Analytical")
    ax.set_xlabel("x (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_transient(result, variable="T", title="Transient solution", analytical_fn=None, ylabel=None):
    times = result["times"]
    idx = st.slider("Time level", 0, len(times) - 1, len(times) - 1)
    vals = result["history"][idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("Time", f"{times[idx]:.6g} s")
    c2.metric("Δx", f"{result['dx']:.6g} m")
    c3.metric("Fourier number", f"{result.get('Fo', float('nan')):.6g}")
    analytical = analytical_fn(times[idx]) if analytical_fn is not None else None
    table = df_1d(result["x"], vals, variable, analytical)
    st.dataframe(table, hide_index=True)
    download_csv(table, f"{variable.lower()}_transient.csv")
    if analytical is not None:
        st.metric("Max absolute error vs analytical", f"{np.max(np.abs(vals - analytical)):.6g}")
    render_plot(result["x"], vals, analytical, ylabel=ylabel or variable, title=title)


def show_setup(title, equation, boundaries, grid):
    st.subheader(title)
    st.markdown("**Governing equation**")
    st.latex(equation)
    st.markdown("**Boundary / initial conditions**")
    for line in boundaries:
        st.write(f"- {line}")
    st.markdown("**Computational domain / grid**")
    for line in grid:
        st.write(f"- {line}")


st.title("🧮 CFD Classroom Solver")
st.caption("ME46220 / ME46707 · FDM + FVM numerical solver for the problem families used in the uploaded course material")

with st.sidebar:
    st.header("Navigate")
    section = st.radio(
        "Section",
        label_visibility="collapsed",
        options=["Home", "Course Problems", "Custom FDM", "Custom FVM", "PDE + Method Guide", "Verification"],
    )
    st.divider()
    st.caption("Scope is intentionally limited to the numerical methods needed for the uploaded MST-1, MST-2 and assignment problems.")


if section == "Home":
    st.subheader("What this tool is")
    st.write(
        "A small classroom CFD solver. Select a course problem or enter a new problem of the same numerical type. "
        "The tool shows the setup, numerical result, residual, matrix where applicable, and a plot."
    )
    cols = st.columns(4)
    cols[0].metric("FDM", "3 solver modes")
    cols[1].metric("FVM", "4 solver modes")
    cols[2].metric("Course templates", "11 numerical")
    cols[3].metric("Validation", "residual + tests")
    st.subheader("Included scope")
    st.markdown(
        """
        **Included:** 1D steady/transient diffusion with FDM, 1D steady/transient diffusion with FVM,
        steady convection-diffusion with central/upwind differencing, transient convection-diffusion
        with fully implicit treatment, and the Mach-number PDE classifier used in MST-1.

        **Not included:** 2D/3D meshing, SIMPLE, turbulence models, CAD geometry, or natural-language
        question parsing. Those are outside the limited MST-1/MST-2 solver scope.
        """
    )
    st.info("Use **Course Problems** first to reproduce the uploaded assignment/MST problem types. Use **Custom FDM/FVM** for new numerical values.")


elif section == "Course Problems":
    st.subheader("Course / MST problem library")
    choice = st.selectbox(
        "Select a problem",
        [
            "Assignment-1 Q2 · steady FDM",
            "Assignment-1 Q3 · transient FDM",
            "Assignment-2 Q1 · steady FVM slab",
            "Assignment-2 Q2 · steady FVM fin",
            "Assignment-2 Q3 · transient FVM slab",
            "MST-1 Q1 · CFD-code structure",
            "MST-1 Q2 · PDE classification",
            "MST-1 Q3 · steady FDM",
            "MST-1 Q4 · transient implicit FDM",
            "MST-2 Q1 · steady FVM",
            "MST-2 Q2 · transient explicit FVM",
            "MST-2 Q3 · steady convection-diffusion FVM",
            "MST-2 Q4 · transient convection-diffusion FVM",
        ],
    )

    if choice.startswith("MST-1 Q1"):
        st.subheader("Basic CFD code structure")
        st.code("Preprocessor → Solver → Convergence check → Postprocessor → Validation")
        st.write("Preprocessing: domain/grid, physical properties, boundary conditions.")
        st.write("Solver: discretization, solution method, initialization, iterative/direct solution.")
        st.write("Postprocessing: plots, data extraction, validation.")

    elif choice.startswith("MST-1 Q2"):
        M = st.number_input("Free-stream Mach number M∞", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        D = 4.0 * (2.0 * M * M - 1.0)
        kind = "Elliptic" if D < 0 else ("Parabolic" if abs(D) < 1e-12 else "Hyperbolic")
        show_setup(
            "2D velocity-potential PDE",
            r"A\phi_{xx}+B\phi_{xy}+C\phi_{yy}=0",
            ["A = 1 − M∞²", "B = −2M∞²", "C = 1 − M∞²"],
            ["D = B² − 4AC = 4(2M∞² − 1)"],
        )
        st.metric("Discriminant D", f"{D:.6f}")
        st.metric("Mathematical behaviour", kind)

    elif choice.startswith("Assignment-1 Q2"):
        show_setup(
            "Assignment-1 Q2 · steady FDM",
            r"k\frac{d^2T}{dx^2}+(1000-5T)=0",
            ["T(0) = 300 °C", "q''(L) = 1000 W/m²"],
            ["L = 0.20 m", "5 nodes"],
        )
        r = solve_steady_1d_heat(.2, 5, 1, 1000, 5, 300, 1000)
        analytical = 232*np.exp(np.sqrt(5)*r["x"]) - 132*np.exp(-np.sqrt(5)*r["x"]) + 200
        table = df_1d(r["x"], r["T"], "Numerical T (°C)", analytical)
        st.dataframe(table, hide_index=True)
        download_csv(table, "assignment1_q2.csv")
        st.metric("Max absolute error", f"{np.max(np.abs(r['T']-analytical)):.6g} °C")
        st.metric("Matrix residual", f"{max_residual(r['A'], r['T'], r['b']):.3e}")
        T, dx, kk = r["T"], r["dx"], 1.0
        q_fwd = -kk * (T[1] - T[0]) / dx                      # 1st-order (forward difference)
        q_2nd = -kk * (-3 * T[0] + 4 * T[1] - T[2]) / (2 * dx)  # 2nd-order (one-sided)
        st.markdown("**Heat flux at x = 0**  (q = −k dT/dx)")
        st.dataframe(pd.DataFrame({
            "Method": ["Analytical", "Forward difference (1st order)", "One-sided (2nd order)"],
            "q(0) (W/m²)": [A1_Q2_Q0, q_fwd, q_2nd],
            "Error (%)": [0.0, abs(q_fwd - A1_Q2_Q0) / abs(A1_Q2_Q0) * 100, abs(q_2nd - A1_Q2_Q0) / abs(A1_Q2_Q0) * 100],
        }), hide_index=True)
        render_matrix(r["A"], r["b"])
        render_plot(r["x"], r["T"], analytical, ylabel="T (°C)", title="Assignment-1 Q2")

    elif choice.startswith("Assignment-1 Q3"):
        show_setup(
            "Assignment-1 Q3 · transient FDM",
            r"\frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2}",
            ["T(x,0) = 30 °C", "T(-b,t) = T(b,t) = 250 °C"],
            ["2b = 10 mm", "k = 0.25 W/m·K", "ρ = 1300 kg/m³", "c = 2000 J/kg·K"],
        )
        scheme = st.radio("Scheme", ["explicit", "implicit"], horizontal=True)
        common = dict(length=.01, nodes=11, k=.25, rho=1300, cp=2000, dt=2.5, total_time=10,
                      initial_temperature=30, left_bc={"kind":"dirichlet","value":250}, right_bc={"kind":"dirichlet","value":250})
        r = solve_transient_1d_heat_explicit_general(**common) if scheme == "explicit" else solve_transient_1d_heat_implicit_general(**common)
        st.caption("Analytical series uses x measured from the slab centre; the grid here runs from the left face (x = 0) to the right face (x = 2b).")
        render_transient(r, variable="T (°C)", title=f"Assignment-1 Q3 · {scheme}",
                         analytical_fn=lambda t: a1_q3_exact(r["x"] - 0.005, t), ylabel="T (°C)")

    elif choice.startswith("Assignment-2 Q1"):
        show_setup(
            "Assignment-2 Q1 · steady FVM slab",
            r"\frac{d}{dx}\left(k\frac{dT}{dx}\right)+\dot q=0",
            ["Both sides convect to T∞ = 20 °C", "h = 20 W/m²·°C"],
            ["2a = 0.20 m", "4 control volumes", "k = 2 W/m·°C", "q''' = 20,000 W/m³"],
        )
        r = solve_steady_diffusion(.2, 4, 2, 20000, 0, convection(20, 20), convection(20, 20))
        analytical = a2_q1_exact(r["x"] - 0.1)   # x = 0 at the slab centre (x = 0.1 m here)
        table = df_1d(r["x"], r["phi"], "Numerical T (°C)", analytical)
        st.caption("FVM reports cell-center coordinates (measured from the left face). The analytical T = 170 − 5000x² "
                   "uses x from the slab centre, so it is evaluated at x − 0.1 m. The assignment table lists boundary "
                   "locations; these are not the same points as FVM cell centres.")
        st.dataframe(table, hide_index=True)
        download_csv(table, "assignment2_q1.csv")
        st.metric("Max cell-center error", f"{np.max(np.abs(r['phi']-analytical)):.6g} °C")
        st.metric("Matrix residual", f"{max_residual(r['A'], r['phi'], r['b']):.3e}")
        render_matrix(r["A"], r["b"])
        render_plot(r["x"], r["phi"], analytical, ylabel="T (°C)", title="Assignment-2 Q1")

    elif choice.startswith("Assignment-2 Q2"):
        width, thick, L = .2, .002, .02
        A = width*thick; P = 2*(width+thick); h, k, Tinf, Tw = 15, 45, 25, 225
        B = h*P/A
        show_setup(
            "Assignment-2 Q2 · rectangular fin",
            r"\frac{d}{dx}\left(kA\frac{dT}{dx}\right)-hP(T-T_\infty)=0",
            ["T(0) = 225 °C", "Tip is treated as insulated for the tanh heat-loss expression"],
            ["L = 0.02 m", "4 control volumes", "h = 15 W/m²·°C", "k = 45 W/m·°C"],
        )
        r = solve_steady_diffusion(L, 4, k, B*Tinf, B, dirichlet(Tw), neumann_flux(0))
        m = math.sqrt(h*P/(k*A))
        analytical = Tinf + (Tw-Tinf)*np.cosh(m*(L-r["x"])) / np.cosh(m*L)
        table = df_1d(r["x"], r["phi"], "Numerical T (°C)", analytical)
        st.dataframe(table, hide_index=True)
        download_csv(table, "assignment2_q2_fin.csv")
        st.warning("The uploaded PDF typesets cos/cos for the fin temperature profile, while its tanh heat-loss formula corresponds to the standard insulated-tip cosh/cosh solution. This interface uses the internally consistent form for validation.")
        Q_num = (2.0 * k * A / r["dx"]) * (Tw - r["phi"][0])   # base conduction through half a cell
        Q_ex = a2_q2_heat_exact(L, h, P, k, A, Tw, Tinf)
        st.metric("Max cell-center error", f"{np.max(np.abs(r['phi'] - analytical)):.6g} °C")
        st.markdown("**Heat loss from the fin**")
        st.dataframe(pd.DataFrame({"Method": ["Analytical  √(hPkA)(Tw−T∞)tanh(mL)", "FVM (conduction at fin base)"],
                                   "Q (W)": [Q_ex, Q_num],
                                   "Error (%)": [0.0, abs(Q_num - Q_ex) / Q_ex * 100]}), hide_index=True)
        render_plot(r["x"], r["phi"], analytical, ylabel="T (°C)", title="Assignment-2 Q2 · Fin")

    elif choice.startswith("Assignment-2 Q3"):
        show_setup(
            "Assignment-2 Q3 · transient FVM slab",
            r"\frac{\partial(\rho cT)}{\partial t}=\frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right)",
            ["T(x,0) = 120 °C", "Both surfaces convect to 20 °C"],
            ["2a = 0.20 m", "4 control volumes", "k = 2 W/m·K", "ρ = 1000 kg/m³", "c = 20 J/kg·K"],
        )
        scheme = st.radio("Time scheme", ["explicit", "implicit"], horizontal=True)
        r = solve_transient_diffusion(.2, 4, 2, 20000, .1, .4, 120, 0, 0, convection(20,20), convection(20,20), scheme=scheme)
        st.caption("Analytical series uses x measured from the slab centre; FVM cell centres are measured from the left face, so x − 0.1 m is used.")
        render_transient(r, variable="T (°C)", title=f"Assignment-2 Q3 · {scheme}",
                         analytical_fn=lambda t: a2_q3_exact(r["x"] - 0.1, t), ylabel="T (°C)")

    elif choice.startswith("MST-1 Q3"):
        show_setup(
            "MST-1 Q3 · steady FDM",
            r"k\frac{d^2T}{dx^2}+(1273-T)=0",
            ["T(0) = 373 K", "q''(0.5) = 1000 W/m²"],
            ["L = 0.5 m", "5 nodes", "k = 1 W/m·K"],
        )
        st.warning("The question gives T(0) = 373 K. In the worked solution the first RHS entry is printed as 273 and the "
                   "elimination (292.89 = 273 + 19.89) uses 273, although the final vector is labelled T1 = 373 K. "
                   "The key values 431.86 / 577.31 / 711.5 / 836.5 K therefore correspond to T1 = 273 K, not 373 K. "
                   "This tool solves the problem as stated (373 K), so its interior values differ from the printed key. "
                   "The key also rounds 2.015625 to 2.015, which adds a further ~1–2 K of hand-calculation difference.")
        r = solve_steady_1d_heat(.5, 5, 1, 1273, 1, 373, 1000)
        table = df_1d(r["x"], r["T"], "T (K)")
        st.dataframe(table, hide_index=True)
        download_csv(table, "mst1_q3.csv")
        st.metric("Matrix residual", f"{max_residual(r['A'], r['T'], r['b']):.3e}")
        render_matrix(r["A"], r["b"])
        render_plot(r["x"], r["T"], ylabel="T (K)", title="MST-1 Q3")

    elif choice.startswith("MST-1 Q4"):
        show_setup(
            "MST-1 Q4 · fully implicit FDM",
            r"\frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2}",
            ["T(x,0) = 30 °C", "dT/dx = 0 at x = 0", "T = 250 °C at x = 5 mm"],
            ["Δx = 1 mm", "Δt = 10.40 s", "k = 0.25 W/m·K", "ρ = 1300 kg/m³", "c = 2000 J/kg·K"],
        )
        r = solve_transient_1d_heat_implicit_general(.005, 6, .25, 1300, 2000, 10.4, 10.4, 30,
              {"kind":"neumann","value":0}, {"kind":"dirichlet","value":250})
        render_transient(r, ylabel="T (°C)", title="MST-1 Q4")

    elif choice.startswith("MST-2 Q1"):
        show_setup("MST-2 Q1 · steady FVM", r"\frac{d}{dx}\left(k\frac{dT}{dx}\right)+(1000-5T)=0",
                    ["T(0) = 300 °C", "q''(L) = 1000 W/m²"], ["L = 0.2 m", "4 control volumes", "k = 1 W/m·°C"])
        r = solve_steady_diffusion(.2, 4, 1, 1000, 5, dirichlet(300), neumann_flux(1000))
        table = df_1d(r["x"], r["phi"], "T / φ")
        st.dataframe(table, hide_index=True)
        download_csv(table, "mst2_q1.csv")
        st.metric("Matrix residual", f"{max_residual(r['A'], r['phi'], r['b']):.3e}")
        render_matrix(r["A"], r["b"])

    elif choice.startswith("MST-2 Q2"):
        show_setup("MST-2 Q2 · explicit transient FVM", r"\frac{\partial(\rho cT)}{\partial t}=\frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right)",
                    ["T(x,0) = 200 °C", "T(0,t) = 0 °C", "dT/dx = 0 at x = 0.02 m"], ["L = 0.02 m", "4 control volumes", "k = 10 W/m·K", "ρc = 10×10⁶ J/m³·K"])
        r = solve_transient_diffusion(.02, 4, 10, 10e6, 1, 4, 200, 0, 0, dirichlet(0), neumann_flux(0), scheme="explicit")
        render_transient(r, ylabel="T (°C)", title="MST-2 Q2")

    elif choice.startswith("MST-2 Q3"):
        show_setup("MST-2 Q3 · steady convection-diffusion", r"\frac{d}{dx}(\rho u\phi)=\frac{d}{dx}\left(\Gamma\frac{d\phi}{dx}\right)",
                    ["φ(0)=1", "φ(0.1)=0"], ["L = 0.1 m", "5 control volumes", "ρ = 12", "u = 1 m/s", "Γ = 0.24"])
        scheme = st.radio("Spatial scheme", ["central", "upwind"], horizontal=True)
        r = solve_steady_convection_diffusion(.1, 5, 12, 1, .24, dirichlet(1), dirichlet(0), scheme=scheme)
        st.metric("Cell Peclet number", f"{r['cell_Pe']:.6f}")
        table = df_1d(r["x"], r["phi"], "φ")
        st.dataframe(table, hide_index=True)
        download_csv(table, f"mst2_q3_{scheme}.csv")
        st.metric("Matrix residual", f"{max_residual(r['A'], r['phi'], r['b']):.3e}")
        render_matrix(r["A"], r["b"])
        render_plot(r["x"], r["phi"], ylabel="φ", title=f"MST-2 Q3 · {scheme}")

    else:
        show_setup("MST-2 Q4 · implicit transient convection-diffusion", r"\frac{\partial(\rho\phi)}{\partial t}+\frac{\partial(\rho u\phi)}{\partial x}=\frac{\partial}{\partial x}\left(\Gamma\frac{\partial\phi}{\partial x}\right)",
                    ["φ(x,0)=0", "φ(0,t)=0", "dφ/dx=0 at x=1.5 m"], ["L = 1.5 m", "3 control volumes", "ρ = 1", "u = 2 m/s", "Γ = 0.03"])
        r = solve_transient_convection_diffusion_implicit(1.5, 3, 1, 2, .03, 1, .1, .5, 0, 0, neumann_flux(0), scheme="central")
        render_transient(r, variable="φ", title="MST-2 Q4")
        st.info("With the printed initial and boundary data all equal to zero, the exact/physical solution is the zero field. The solver reproduces that state.")


elif section == "Custom FDM":
    st.subheader("Custom 1D FDM solver")
    mode = st.radio("Problem type", ["Steady heat conduction", "Transient heat conduction"], horizontal=True)
    if mode == "Steady heat conduction":
        c1, c2 = st.columns(2)
        L = c1.number_input("Length L (m)", min_value=1e-6, value=.2)
        N = int(c1.number_input("Nodes", min_value=3, max_value=501, value=5, step=1))
        k = c2.number_input("k", min_value=1e-12, value=1.0)
        A = c2.number_input("Source A in q''' = A − B·T", value=1000.0)
        B = c2.number_input("Source B in q''' = A − B·T", value=5.0)
        TL = c1.number_input("Left temperature", value=300.0)
        qR = c2.number_input("Right heat flux k·dT/dx", value=1000.0)
        if st.button("Solve steady FDM", type="primary"):
            try:
                r = solve_steady_1d_heat(L, N, k, A, B, TL, qR)
                table = df_1d(r["x"], r["T"], "T")
                st.dataframe(table, hide_index=True)
                download_csv(table, "custom_fdm_steady.csv")
                st.metric("Matrix residual", f"{max_residual(r['A'], r['T'], r['b']):.3e}")
                render_matrix(r["A"], r["b"])
                render_plot(r["x"], r["T"], ylabel="T", title="Custom steady FDM")
            except Exception as exc:
                st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        L = c1.number_input("Length L (m)", min_value=1e-6, value=.01, key="fdmtL")
        N = int(c1.number_input("Nodes", min_value=3, max_value=501, value=11, step=1, key="fdmtN"))
        k = c2.number_input("k", min_value=1e-12, value=1.0, key="fdmtk")
        rho = c2.number_input("ρ", min_value=1e-12, value=1000.0, key="fdmtrho")
        cp = c2.number_input("c", min_value=1e-12, value=1000.0, key="fdmtcp")
        dt = c1.number_input("Δt (s)", min_value=1e-10, value=.001, format="%.8g", key="fdmdt")
        total = c2.number_input("Total time (s)", min_value=1e-10, value=.01, format="%.8g", key="fdmtotal")
        T0 = c1.number_input("Initial temperature", value=20.0, key="fdmT0")
        left_kind = c1.selectbox("Left BC", ["Dirichlet", "Neumann"], key="fdmlkind")
        left_val = c1.number_input("Left BC value", value=100.0 if left_kind == "Dirichlet" else 0.0, key="fdmlval")
        right_kind = c2.selectbox("Right BC", ["Dirichlet", "Neumann"], key="fdmrkind")
        right_val = c2.number_input("Right BC value", value=20.0 if right_kind == "Dirichlet" else 0.0, key="fdmrval")
        scheme = c2.selectbox("Time scheme", ["explicit", "implicit"], key="fdmscheme")
        if st.button("Solve transient FDM", type="primary"):
            try:
                bcL = {"kind": left_kind.lower(), "value": left_val}
                bcR = {"kind": right_kind.lower(), "value": right_val}
                fn = solve_transient_1d_heat_explicit_general if scheme == "explicit" else solve_transient_1d_heat_implicit_general
                r = fn(L, N, k, rho, cp, dt, total, T0, bcL, bcR)
                render_transient(r, ylabel="T", title=f"Custom transient FDM · {scheme}")
                if scheme == "explicit":
                    st.write(f"Stability check: Fo = {r['Fo']:.6g} (solver also enforces the positive-coefficient rule; see error message if rejected)")
            except Exception as exc:
                st.error(str(exc))


elif section == "Custom FVM":
    st.subheader("Custom 1D FVM solver")
    mode = st.selectbox("Problem type", ["Steady diffusion", "Transient diffusion", "Steady convection-diffusion", "Transient convection-diffusion"])

    if mode in {"Steady diffusion", "Transient diffusion"}:
        c1, c2 = st.columns(2)
        L = c1.number_input("Length L (m)", min_value=1e-6, value=.2, key="fvml")
        cells = int(c1.number_input("Control volumes", min_value=1, max_value=500, value=4, step=1, key="fvmcells"))
        Gamma = c2.number_input("Diffusion coefficient Γ", min_value=1e-12, value=1.0, key="fvmGamma")
        A = c2.number_input("Source A", value=0.0, key="fvmA")
        B = c2.number_input("Source B in S = A − B·φ", value=0.0, key="fvmB")

        def boundary_widget(prefix):
            kind = st.selectbox(f"{prefix} boundary", ["Dirichlet", "Flux", "Convection"], key=prefix+"kind")
            if kind == "Dirichlet":
                return dirichlet(st.number_input(f"{prefix} value", value=0.0, key=prefix+"value"))
            if kind == "Flux":
                return neumann_flux(st.number_input(f"{prefix} heat flux INTO the domain (W/m²)", value=0.0, key=prefix+"flux"))
            h = st.number_input(f"{prefix} h", min_value=0.0, value=20.0, key=prefix+"h")
            amb = st.number_input(f"{prefix} ambient", value=20.0, key=prefix+"amb")
            return convection(h, amb)

        left = boundary_widget("Left")
        right = boundary_widget("Right")

        if mode == "Steady diffusion":
            if st.button("Solve steady FVM", type="primary"):
                try:
                    r = solve_steady_diffusion(L, cells, Gamma, A, B, left, right)
                    table = df_1d(r["x"], r["phi"], "φ")
                    st.dataframe(table, hide_index=True)
                    download_csv(table, "custom_fvm_steady.csv")
                    st.caption("x is the cell-center coordinate.")
                    st.metric("Matrix residual", f"{max_residual(r['A'], r['phi'], r['b']):.3e}")
                    render_matrix(r["A"], r["b"])
                    render_plot(r["x"], r["phi"], ylabel="φ", title="Custom steady FVM")
                except Exception as exc:
                    st.error(str(exc))
        else:
            c1, c2 = st.columns(2)
            rho_cp = c1.number_input("ρc (storage coefficient)", min_value=1e-12, value=1e6, key="fvmrcp")
            dt = c2.number_input("Δt", min_value=1e-10, value=.01, key="fvmdt")
            total = c2.number_input("Total time", min_value=1e-10, value=.1, key="fvmtotal")
            initial = c1.number_input("Initial φ", value=0.0, key="fvminit")
            scheme = c1.selectbox("Time scheme", ["explicit", "implicit"], key="fvmscheme")
            if st.button("Solve transient FVM", type="primary"):
                try:
                    r = solve_transient_diffusion(L, cells, Gamma, rho_cp, dt, total, initial, A, B, left, right, scheme=scheme)
                    render_transient(r, ylabel="φ", title=f"Custom transient FVM · {scheme}")
                    if scheme == "explicit":
                        st.write(f"Stability check: Fo = {r['Fo']:.6g} (solver also enforces the positive-coefficient rule; see error message if rejected)")
                except Exception as exc:
                    st.error(str(exc))

    else:
        c1, c2 = st.columns(2)
        L = c1.number_input("Length L (m)", min_value=1e-6, value=.1, key="cdL")
        cells = int(c1.number_input("Control volumes", min_value=2, max_value=500, value=5, step=1, key="cdCells"))
        rho = c2.number_input("ρ", min_value=1e-12, value=12.0, key="cdRho")
        u = c2.number_input("u", value=1.0, key="cdU")
        Gamma = c2.number_input("Γ", min_value=1e-12, value=.24, key="cdGamma")
        phiL = c1.number_input("Left φ", value=1.0, key="cdPhiL")
        phiR = c2.number_input("Right φ", value=0.0, key="cdPhiR")
        scheme = c1.selectbox("Spatial scheme", ["central", "upwind"], key="cdScheme")
        left = dirichlet(phiL)
        right = dirichlet(phiR)
        if mode == "Steady convection-diffusion":
            if st.button("Solve steady convection-diffusion", type="primary"):
                try:
                    r = solve_steady_convection_diffusion(L, cells, rho, u, Gamma, left, right, scheme=scheme)
                    table = df_1d(r["x"], r["phi"], "φ")
                    st.metric("Cell Peclet number", f"{r['cell_Pe']:.6f}")
                    st.dataframe(table, hide_index=True)
                    download_csv(table, f"custom_convection_diffusion_{scheme}.csv")
                    st.metric("Matrix residual", f"{max_residual(r['A'], r['phi'], r['b']):.3e}")
                    render_matrix(r["A"], r["b"])
                    render_plot(r["x"], r["phi"], ylabel="φ", title=f"Custom convection-diffusion · {scheme}")
                except Exception as exc:
                    st.error(str(exc))
        else:
            rho_cp = c1.number_input("ρ for transient storage", min_value=1e-12, value=1.0, key="cdRhoCp")
            dt = c2.number_input("Δt", min_value=1e-10, value=.1, key="cdDt")
            total = c2.number_input("Total time", min_value=1e-10, value=.5, key="cdTotal")
            initial = c1.number_input("Initial φ", value=0.0, key="cdInit")
            right_flux = neumann_flux(0.0)
            if st.button("Solve transient convection-diffusion", type="primary"):
                try:
                    r = solve_transient_convection_diffusion_implicit(L, cells, rho, u, Gamma, rho_cp, dt, total, initial, phiL, right_flux, scheme=scheme)
                    render_transient(r, variable="φ", title=f"Custom transient convection-diffusion · {scheme}")
                    st.metric("Cell Peclet number", f"{r['cell_Pe']:.6f}")
                except Exception as exc:
                    st.error(str(exc))


elif section == "PDE + Method Guide":
    tabs = st.tabs(["PDE classifier", "Method guide", "Boundary conditions", "Important numerical checks"])
    with tabs[0]:
        M = st.number_input("M∞", min_value=0.0, max_value=10.0, value=.5, step=.1)
        D = 4.0*(2.0*M*M-1.0)
        kind = "Elliptic" if D < 0 else ("Parabolic" if abs(D) < 1e-12 else "Hyperbolic")
        st.latex(r"D=B^2-4AC=4(2M_\infty^2-1)")
        st.metric("D", f"{D:.6f}")
        st.metric("Classification", kind)
    with tabs[1]:
        st.markdown("### FDM")
        st.write("Replace derivatives at grid nodes by finite-difference formulas, producing algebraic equations for nodal values.")
        st.markdown("### FVM")
        st.write("Integrate the conservation equation over each control volume and balance fluxes across its faces.")
        st.markdown("### Explicit")
        st.write("The new time level is computed directly from known values at the old time level.")
        st.markdown("### Fully implicit")
        st.write("The new time level appears in the algebraic system, so a matrix system is solved at each time step.")
        st.markdown("### Central vs upwind")
        st.write("Both are spatial convection-diffusion discretizations included here because central and upwind are explicitly in Unit IV and central appears in MST-2 Q3.")
    with tabs[2]:
        st.write("Dirichlet: prescribe the variable value.")
        st.write("Neumann: prescribe a derivative/flux.")
        st.write("Robin/convection: connect boundary flux to the boundary and ambient values through h.")
    with tabs[3]:
        st.write("Explicit diffusion: Fo ≤ 0.5 in interior cells; cells next to a fixed-temperature boundary need Fo ≤ 1/3 (positive-coefficient rule). The solver checks this.")
        st.write("Every linear solve: check the algebraic residual ||A·x-b||∞.")
        st.write("FVM coordinates are cell centers, so they should not be confused with boundary nodes in an FDM table.")
        st.write("When an uploaded PDF is internally inconsistent, the discrepancy is flagged instead of silently hidden.")


else:
    st.subheader("Verification")
    st.write("The repository contains automated checks for the uploaded problem families plus several new, non-copied parameter sets.")
    st.code("python course_coverage_test.py\npython run_all_tests.py")
    st.write("The final submission should only be deployed after these tests pass locally.")
    st.info("The verification report is part of the project package so the professor can see how the solver was checked.")

# Minimal attribution footer. Kept intentionally small so it stays unobtrusive.
st.markdown(
    """
    <style>
    .made-by {
        position: fixed;
        right: 16px;
        bottom: 8px;
        font-size: 11px;
        color: #6f6f6f;
        opacity: 0.85;
        z-index: 999;
        pointer-events: none;
        font-weight: 500;
    }
    </style>
    <div class="made-by">Yash Patel • Mechanical Engineering</div>
    """,
    unsafe_allow_html=True,
)
