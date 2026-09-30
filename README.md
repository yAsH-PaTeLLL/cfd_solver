# CFD Classroom Solver

A small, transparent Python + Streamlit numerical solver aligned to the CFD problem families in the uploaded ME46220/ME46707 course material.

## Purpose

The project is intended as a reusable classroom tool for the instructor's MST-1/MST-2-style numerical problems. It is deliberately **not** a general-purpose CFD package.

## Final scope

### FDM
- 1D steady heat conduction with a linear source and mixed temperature/heat-flux boundaries
- 1D transient heat conduction using explicit FTCS
- 1D transient heat conduction using fully implicit time marching

### FVM diffusion
- 1D steady diffusion/heat conduction with linear sources
- 1D transient diffusion using explicit or fully implicit time marching
- convection (Robin) boundary treatment

### FVM convection-diffusion
- 1D steady convection-diffusion using central differencing
- 1D steady convection-diffusion using upwind differencing
- 1D transient convection-diffusion using fully implicit time marching

### Supporting tools
- Mach-number PDE classification used in MST-1
- direct Gaussian elimination with partial pivoting
- matrix residual checks
- analytical-solution comparison where an analytical form is provided/verified
- plots and CSV export
- course/MST problem presets
- custom parameter entry for new similar 1D problems

## Intentionally excluded

The bonus version does not attempt to implement all advanced syllabus topics. In particular, it does not include 2D/3D general meshing, SIMPLE, pressure-velocity correction, turbulence models, k-epsilon, hybrid/power-law/QUICK schemes, or natural-language parsing. Those are outside the focused MST-1/MST-2 solver requirement.

## Files

- `fdm.py` - finite-difference algorithms
- `fvm.py` - finite-volume algorithms
- `utils.py` - linear solve, residual, comparison and plotting helpers
- `analytical.py` - closed-form solutions quoted in the assignments (used by the app and the tests)
- `examples.py` - worked course examples
- `app.py` - final Streamlit classroom interface
- `run_all_tests.py` - numerical verification suite
- `course_coverage_test.py` - assignment/MST coverage + new similar-question tests
- `FINAL_SCOPE.md` - locked scope decision
- `verification_notes.md` - documented source inconsistencies and numerical checks

## Local use

Create/activate a virtual environment, then:

```bash
python -m pip install -r requirements.txt
python run_all_tests.py
python course_coverage_test.py
streamlit run app.py
```

## Verification philosophy

The numerical code is checked using a combination of:

1. instructor-reported numerical results where the uploaded PDF is internally consistent,
2. independently evaluated analytical solutions,
3. comparison against `numpy.linalg.solve` for the same linear system,
4. direct residual checks `A @ x - b`,
5. grid-refinement checks,
6. explicit-scheme stability checks,
7. new, non-copied parameter sets.

Where an uploaded PDF contains an internal inconsistency, the tool flags it instead of silently modifying the question.

## Known source inconsistencies documented in the project

- MST-1 Q3: the question states 373 K at x=0, but the worked solution's first RHS entry is printed as 273 and its elimination (292.89 = 273 + 19.89) uses 273. The printed key (431.86, 577.31, 711.54, 836.54 K) is what the instructor's own matrix gives for T1 = 273 K (verified in `run_all_tests.py`). The solver follows the stated question (373 K), so its interior values differ from the printed key. The key also rounds 2.015625 to 2.015.
- Assignment-2 Q2: the fin temperature expression is typeset with cos/cos while its tanh heat-loss expression is consistent with the standard insulated-tip cosh/cosh form; the cosh/cosh form is used.
- Assignment-2 Q1/Q3: the supplied analytical expressions use x measured from the slab centre, while the FVM solver reports cell centres measured from the left face; the comparison shifts coordinates explicitly.
- Assignment tables list 5 node rows (x = 0 ... 0.2 m) for FVM problems that ask for 4 control volumes. This tool uses cell-centred control volumes (4 values); it does not reproduce a 5-row boundary-node table. Confirm the intended FVM grid convention with the instructor.
- Assignment-1 Q3 and Assignment-2 Q3 do not specify the grid spacing or time step; the course-problem pages use stated defaults (11 nodes, dt = 2.5 s, and 4 CVs, dt = 0.1 s).

## Sign convention for flux boundaries

Flux values are **heat flux INTO the domain** at either end (for the right end this equals k dT/dx, matching the instructor's MST-1 Q3 treatment, k dT/dx = 1000).
