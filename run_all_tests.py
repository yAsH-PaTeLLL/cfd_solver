"""Cross-check the full CFD solver with course cases and independent references."""
from __future__ import annotations

import numpy as np
from examples import mst1_q3_fdm, mst1_q4_transient, mst2_q1_fvm, assignment2_q1_fvm_robin, mst2_q3_convection_diffusion, mst2_q4_transient_cd
from utils import gaussian_elimination
from fvm import solve_steady_diffusion, solve_steady_convection_diffusion, dirichlet, convection


def check_close(name, got, ref, tol):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))
    print(f"{name:<55} max error = {err:.8e}")
    assert err <= tol, f"{name}: error {err} > {tol}"


def main():
    print("="*80)
    print("CFD SOLVER - FULL CROSS-CHECK SUITE")
    print("="*80)

    # Linear algebra cross-check.
    rng = np.random.default_rng(20260930)
    A = rng.normal(size=(8,8))
    A += 5*np.eye(8)
    b = rng.normal(size=8)
    check_close("Gaussian elimination vs numpy.linalg.solve", gaussian_elimination(A,b), np.linalg.solve(A,b), 1e-10)

    # MST-1 Q3 question-as-printed FDM: exact analytical solution.
    r, analytical, ea, ep = mst1_q3_fdm()
    # Five nodes is intentionally coarse, so the finite-difference solution
    # should agree with the analytical field only to discretization error.
    check_close("MST-1 Q3 FDM vs analytical solution (5-node grid)", r["T"], analytical, 12.0)
    check_close("MST-1 Q3 matrix residual", r["A"] @ r["T"], r["b"], 1e-10)

    # MST-1 Q4 implicit: instructor reported first time level.
    implicit, _ = mst1_q4_transient()
    expected = np.array([36.47,36.47,42.94,62.36,114.12,250.0])
    check_close("MST-1 Q4 implicit first step vs rounded instructor answer", implicit["history"][1], expected, 0.01)
    check_close("MST-1 Q4 implicit matrix residual", implicit["matrix"] @ implicit["history"][1], np.array([0.0,30.0,30.0,30.0,30.0,250.0]), 1e-10)

    # The MST-1 time step gives Fo≈1, which is intentionally too large for
    # standard explicit FTCS diffusion. Verify that the solver rejects it.
    from fdm import solve_transient_1d_heat_explicit
    try:
        solve_transient_1d_heat_explicit(0.005, 6, 0.25, 1300.0, 2000.0, 10.40, 10.40, 30.0, right_temperature=250.0)
    except ValueError:
        print(f"{'Explicit FTCS stability guard at Fo=1':<55} PASS")
    else:
        raise AssertionError("Explicit FTCS stability guard did not reject Fo=1")

    # MST-2 Q1 FVM.
    r_fvm = mst2_q1_fvm()
    check_close("MST-2 Q1 FVM matrix residual", r_fvm["A"] @ r_fvm["phi"], r_fvm["b"], 1e-10)
    # Independent analytical solution for k*T'' + (1000-5T)=0 with
    # T(0)=300 and k*T'(L)=1000.  FVM uses 4 cell centres, so a small
    # discretization error is expected.
    import math
    sf = math.sqrt(5.0)
    MM = np.array([[1.0, 1.0], [math.exp(sf*0.2), -math.exp(-sf*0.2)]])
    cc = np.linalg.solve(MM, np.array([300.0-200.0, 1000.0/sf]))
    exact_mst2 = 200.0 + cc[0]*np.exp(sf*r_fvm["x"]) + cc[1]*np.exp(-sf*r_fvm["x"])
    check_close("MST-2 Q1 FVM vs independent analytical solution", r_fvm["phi"], exact_mst2, 0.30)
    print("MST-2 Q1 FVM cell-center solution:", np.round(r_fvm["phi"], 6))

    # Assignment-2 Q1 FVM with stated physical geometry.
    robin, exact_robin = assignment2_q1_fvm_robin()
    # Four control volumes are intentionally coarse; compare against the exact
    # solution at cell centres as a discretization check, not an identity.
    check_close("Assignment-2 Q1 FVM vs exact solution at cell centres", robin["phi"], exact_robin, 4.0)

    # MST-2 Q3 central/upwind; global Pe = rho*u*L/Gamma = 5, cell Pe = 1.
    central, upwind, exact_cd = mst2_q3_convection_diffusion()
    print(f"Cell Peclet number for MST-2 Q3 = {central['cell_Pe']:.6f}")
    check_close("MST-2 Q3 central FVM matrix residual", central["A"] @ central["phi"], central["b"], 1e-10)
    check_close("MST-2 Q3 upwind FVM matrix residual", upwind["A"] @ upwind["phi"], upwind["b"], 1e-10)
    print("Central solution:", np.round(central["phi"], 8))
    print("Upwind  solution:", np.round(upwind["phi"], 8))
    print("Exact-at-cell-centers:", np.round(exact_cd, 8))
    cd_refine = []
    for n in (5, 10, 20):
        rr = solve_steady_convection_diffusion(0.1, n, 12.0, 1.0, 0.24, dirichlet(1.0), dirichlet(0.0), scheme='central')
        xx = rr["x"]
        exact_n = 1.0 - (np.exp(5.0 * xx / 0.1) - 1.0) / (np.exp(5.0) - 1.0)
        cd_refine.append(float(np.max(np.abs(rr["phi"] - exact_n))))
    print("Central convection-diffusion grid-refinement max errors:", [f"{e:.6f}" for e in cd_refine])
    assert cd_refine[1] < cd_refine[0] and cd_refine[2] < cd_refine[1]

    # MST-2 Q4 transient CD.
    transient_cd = mst2_q4_transient_cd()
    print("MST-2 Q4 transient CD first two states:")
    print(np.round(transient_cd["history"][:2], 8))
    check_close("MST-2 Q4 transient CD initial condition", transient_cd["history"][0], np.zeros(3), 1e-12)


    # Grid-refinement sanity checks: discretization error should decrease.
    from fdm import solve_steady_1d_heat
    ref_errors = []
    for n in (5, 9, 17):
        rr = solve_steady_1d_heat(0.5, n, 1.0, 1273.0, 1.0, 373.0, 1000.0)
        xx = rr["x"]
        # exact solution, same as the Q3 cross-check above
        import math
        Mx = np.array([[1.0, 1.0], [math.exp(0.5), -math.exp(-0.5)]])
        cc = np.linalg.solve(Mx, np.array([373.0-1273.0, 1000.0]))
        aa = cc[0]*np.exp(xx) + cc[1]*np.exp(-xx) + 1273.0
        ref_errors.append(float(np.max(np.abs(rr["T"]-aa))))
    print("FDM grid-refinement max errors (5,9,17 nodes):", [f"{e:.6f}" for e in ref_errors])
    assert ref_errors[1] < ref_errors[0] and ref_errors[2] < ref_errors[1]

    # FVM diffusion refinement check for the stated Assignment-2 physics.
    fvm_ref = []
    for n in (4, 8, 16):
        rr = solve_steady_diffusion(0.2, n, 2.0, 20000.0, 0.0, convection(20.0,20.0), convection(20.0,20.0))
        exact_rr = 120.0 + 1000.0*rr["x"] - 5000.0*rr["x"]**2
        fvm_ref.append(float(np.max(np.abs(rr["phi"]-exact_rr))))
    print("FVM diffusion grid-refinement max errors (4,8,16 CVs):", [f"{e:.6f}" for e in fvm_ref])
    assert fvm_ref[1] < fvm_ref[0] and fvm_ref[2] < fvm_ref[1]

    # Stable explicit FDM run at Fo=0.5.
    from fdm import solve_transient_1d_heat_explicit
    stable = solve_transient_1d_heat_explicit(0.005, 6, 0.25, 1300.0, 2000.0, 5.20, 5.20, 30.0, right_temperature=250.0)
    print(f"Explicit FDM stable test at Fo={stable['Fo']:.6f} -> PASS")
    assert abs(stable["Fo"] - 0.5) < 1e-12
    assert np.all(np.isfinite(stable["history"]))

    # Fully implicit transient FDM refinement against the separated analytical
    # series for the mixed Neumann/Dirichlet problem.
    import math
    from fdm import solve_transient_1d_heat_implicit
    fine_errors = []
    L = 0.005; k = 0.25; rho = 1300.0; cp = 2000.0; t_end = 10.4
    alpha = k/(rho*cp)
    for n_nodes, dt_i in ((11,1.04),(21,0.26),(51,0.052)):
        rr = solve_transient_1d_heat_implicit(L,n_nodes,k,rho,cp,dt_i,t_end,30.0,right_temperature=250.0)
        xx = rr["x"]
        theta = np.zeros_like(xx)
        for nn in range(250):
            lam=(nn+0.5)*math.pi
            theta += (2.0*math.sin(lam)/lam)*np.cos(lam*xx/L)*math.exp(-alpha*(lam/L)**2*t_end)
        exact_t = 250.0 - 220.0*theta
        fine_errors.append(float(np.max(np.abs(rr["history"][-1]-exact_t))))
    print("Implicit transient FDM refinement max errors:", [f"{e:.6f}" for e in fine_errors])
    assert fine_errors[1] < fine_errors[0] and fine_errors[2] < fine_errors[1]


    # ------------------------------------------------------------------
    # REGRESSION CHECKS (bugs found in the v1.2.1 review; must never return)
    # ------------------------------------------------------------------
    from fvm import solve_transient_convection_diffusion_implicit, solve_transient_diffusion, neumann_flux
    from fdm import solve_transient_1d_heat_implicit_general, solve_transient_1d_heat_explicit_general

    # (a) Steady convection-diffusion must conserve flux across EVERY face (both schemes, both flow directions).
    G, L, N = 0.24, 0.1, 5
    Dd = G / (L / N)
    for sch in ("central", "upwind"):
        for sign in (+1, -1):
            pL, pR = (1.0, 0.0) if sign > 0 else (0.0, 1.0)
            rr = solve_steady_convection_diffusion(L, N, 12.0, sign * 1.0, G, dirichlet(pL), dirichlet(pR), scheme=sch)
            F = 12.0 * sign
            ph = rr["phi"]
            def conv_face(i):          # interior face between cell i and i+1
                if sch == "central":
                    return F * 0.5 * (ph[i] + ph[i + 1])
                return F * (ph[i] if F >= 0 else ph[i + 1])
            fluxes = [ (F * pL if (sch == "central" or F >= 0) else F * ph[0]) - 2 * Dd * (ph[0] - pL) ]
            fluxes += [conv_face(i) - Dd * (ph[i + 1] - ph[i]) for i in range(N - 1)]
            fluxes += [(F * pR if (sch == "central" or F < 0) else F * ph[-1]) - 2 * Dd * (pR - ph[-1])]
            spread = float(np.ptp(fluxes))
            print(f"{'Flux conservation ('+sch+', F'+('>0' if sign>0 else '<0')+')':<55} spread = {spread:.3e}")
            assert spread < 1e-9, f"flux not conserved for {sch}, sign {sign}"

    # (b) Transient convection-diffusion with a zero-gradient outlet must relax to the uniform inlet value.
    for sch in ("central", "upwind"):
        rr = solve_transient_convection_diffusion_implicit(1.5, 3, 1, 2, 0.03, 1, 0.1, 200.0, 0.0, 1.0, neumann_flux(0.0), scheme=sch)
        check_close(f"Transient conv-diff outlet -> uniform inlet value ({sch})", rr["history"][-1], np.ones(3), 1e-9)

    # (c) Implicit FDM with a NON-zero Neumann gradient on the left must match the exact linear steady state.
    kw = dict(length=1.0, nodes=11, k=1, rho=1, cp=1, dt=0.05, total_time=50.0, initial_temperature=0.0,
              left_bc={"kind": "neumann", "value": 50.0}, right_bc={"kind": "dirichlet", "value": 100.0})
    xx = np.linspace(0, 1, 11)
    check_close("Implicit FDM left Neumann (g=50) vs exact", solve_transient_1d_heat_implicit_general(**kw)["history"][-1], 100 + 50 * (xx - 1), 1e-6)
    check_close("Explicit FDM left Neumann (g=50) vs exact", solve_transient_1d_heat_explicit_general(**dict(kw, dt=0.004, total_time=20.0))["history"][-1], 100 + 50 * (xx - 1), 1e-3)

    # (d) Explicit FVM: cell next to a Dirichlet boundary needs Fo <= 1/3 (MST-2 Q2 -> dt <= 8.33 s).
    ok = solve_transient_diffusion(.02, 4, 10, 10e6, 8.0, 80.0, 200, 0, 0, dirichlet(0), neumann_flux(0), scheme="explicit")
    assert ok["history"].min() >= -1e-9 and np.all(np.diff(ok["history"][:, 0]) <= 1e-9), "explicit FVM not monotone at Fo<1/3"
    print(f"{'Explicit FVM at Fo=0.32 is monotone (no oscillation)':<55} PASS")
    for bad_dt in (10.0, 12.5):
        try:
            solve_transient_diffusion(.02, 4, 10, 10e6, bad_dt, bad_dt * 2, 200, 0, 0, dirichlet(0), neumann_flux(0), scheme="explicit")
        except ValueError:
            print(f"{'Explicit FVM guard rejects dt='+str(bad_dt)+' (Fo>1/3 at Dirichlet cell)':<55} PASS")
        else:
            raise AssertionError(f"explicit FVM accepted oscillatory dt={bad_dt}")

    # (e) The instructor's MST-1 Q3 printed values are reproduced by HIS matrix when the first RHS entry is 273 (not 373).
    Aprof = np.array([[1,0,0,0,0],[-1,2.015,-1,0,0],[0,-1,2.015,-1,0],[0,0,-1,2.015,-1],[0,0,0,-1,1]], float)
    key = np.array([431.86, 577.31, 711.54, 836.54])
    got273 = np.linalg.solve(Aprof, np.array([273, 19.89, 19.89, 19.89, 125.0]))[1:]
    check_close("MST-1 Q3 printed key == instructor matrix with T1=273", got273, key, 0.03)
    got373 = np.linalg.solve(Aprof, np.array([373, 19.89, 19.89, 19.89, 125.0]))[1:]
    assert np.max(np.abs(got373 - key)) > 50, "key unexpectedly matches 373"
    print(f"{'MST-1 Q3 printed key does NOT match T1=373 (source typo)':<55} PASS")

    # (f) Shared analytical module: Assignment-2 Q1 comparison must use x measured from the slab centre.
    from analytical import a2_q1_exact
    rr = solve_steady_diffusion(0.2, 4, 2.0, 20000.0, 0.0, convection(20.0, 20.0), convection(20.0, 20.0))
    err = float(np.max(np.abs(rr["phi"] - a2_q1_exact(rr["x"] - 0.1))))
    print(f"{'Assignment-2 Q1 error with centred coordinate':<55} {err:.4f} C")
    assert err < 4.0

    print("\nALL CROSS-CHECKS PASSED.")


if __name__ == "__main__":
    main()
