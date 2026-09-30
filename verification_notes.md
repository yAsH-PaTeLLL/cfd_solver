# Verification notes

## 1. MST-1 Q3 inconsistency
The uploaded MST-1 question states `T(x=0)=373 K`. The worked solution prints a first RHS matrix entry of `273`, its first elimination step (`292.89 = 273 + 19.89`) uses 273, and yet the final vector is labelled `T1 = 373 K`. The printed interior values (431.86, 577.31, 711.54, 836.54 K) are exactly what his matrix gives for 273 (test `MST-1 Q3 printed key == instructor matrix with T1=273`); with 373 they would be about 527.7, 670.3, 803.2, 928.2 K (his rounded coefficients) or 526.5, 668.3, 800.7, 925.7 K (exact coefficients, this solver).

The solver treats the question statement as authoritative. Expect a mismatch with the printed key and raise it with the instructor.

## 2. MST-1 Q4
The instructor's fully implicit derivation uses `Fo ≈ 1`, `dx=1 mm`, and `dt=10.40 s`, leading to the matrix with interior pattern `[-1, 3, -1]`. The solver reproduces the rounded first-time-level values `[36.47, 36.47, 42.94, 62.36, 114.12, 250] °C` within rounding error.

## 3. FVM convention
The uploaded current course materials do not include a solved FVM example showing the instructor's exact control-volume boundary notation. The FVM implementation therefore uses the standard cell-centred 1D finite-volume formulation with half-cell boundary distances, consistent with standard NPTEL finite-volume derivations. This is explicitly stated in `fvm.py` rather than pretending the professor's exact private convention was observed.

## 4. Assignment-2 Q1 coordinate inconsistency
The assignment states a slab width `2a=0.2 m` with `a=0.1 m`, but gives an analytical expression `T(x)=170-5000x^2` and a table ranging from `x=0` to `0.2 m`. The expression is the usual half-slab form when x is measured from the symmetry plane and reaches the physical boundary at x=0.1 m. For verification, the example uses the physically stated slab `[0,0.2] m` and the equivalent symmetric-coordinate solution `T=120+1000x-5000x^2`.

## 5. Review fixes (v1.3)
Bugs found in v1.2.1 and corrected; each has a regression test in `run_all_tests.py`.

1. **Steady/transient convection-diffusion, Dirichlet boundary rows.** The east-boundary row used the central-scheme term for upwind and dropped the convective term for negative velocity. Upwind with F>0 was not flux-conservative (east-face flux 18.4 vs 12.3 elsewhere). Central F>0 (the MST-2 Q3 case) was already correct and is unchanged; upwind results for MST-2 Q3 changed to [0.98571, 0.94286, 0.85714, 0.68571, 0.34286].
2. **Transient convection-diffusion, zero-gradient/outflow boundary.** The last row subtracted F, so any non-zero inlet diverged. Now relaxes to the uniform inlet value.
3. **Implicit FDM, left Neumann with non-zero gradient.** RHS sign was reversed (T0 - T1 = -g dx). Zero-gradient cases (all course problems) were unaffected.
4. **Explicit FVM stability.** The Fo <= 0.5 guard allowed oscillatory solutions next to a Dirichlet/convective boundary. The guard now applies the positive-coefficient rule (Fo <= 1/3 at a Dirichlet cell; MST-2 Q2: dt <= 8.33 s).
5. **App.** Four pages crashed (`render_transient` did not accept `ylabel`): MST-1 Q4, MST-2 Q2, custom transient FDM and FVM. The Assignment-2 Q1 page compared against the analytical profile with the wrong x origin (showed ~128 C "error"; true error 3.1 C). Added: heat flux at x=0 (Assignment-1 Q2), fin heat loss (Assignment-2 Q2), analytical columns for the transient pages (Assignment-1 Q3, Assignment-2 Q3). Flux-boundary input was labelled "outward"; it is heat flux into the domain. Removed deprecated `use_container_width`, empty widget label and stray quote characters on the Home page.
