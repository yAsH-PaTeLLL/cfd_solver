# Final Scope Decision

## Required for the instructor bonus

The tool is intentionally centered on the numerical problem families appearing in the uploaded Assignment-1, Assignment-2, MST-1 and MST-2 material:

- PDE classification used in MST-1
- 1D steady FDM heat conduction
- 1D transient FDM heat conduction: explicit and fully implicit
- 1D steady FVM diffusion/heat conduction
- 1D transient FVM diffusion/heat conduction: explicit and fully implicit
- 1D steady convection-diffusion FVM: central and upwind
- 1D transient convection-diffusion FVM: fully implicit

## Not included in the bonus version

The following syllabus topics are outside the deliberately limited MST-1/MST-2 solver scope:

- 2D/3D general-purpose meshing
- Staggered-grid flow-field calculation
- SIMPLE pressure-velocity coupling
- turbulence models
- k-epsilon models
- hybrid/power-law/QUICK schemes
- natural-language parsing of arbitrary written questions
- CAD geometry

These can be future extensions, but they are not prerequisites for the current submission and would increase the risk of introducing bugs without improving the required MST-1/MST-2 coverage.

## Final classroom interface

The browser app contains:

1. Home/scope page
2. Course problem library
3. Custom 1D FDM solver
4. Custom 1D FVM solver
5. PDE classifier + method guide
6. Verification page

For solved numerical problems it provides, where applicable:

- problem setup
- governing equation
- boundary/initial conditions
- computational grid/control volumes
- numerical table
- matrix system
- algebraic residual
- plot
- CSV table download

The interface explicitly distinguishes FDM nodes from FVM cell centers.

## Validation policy

Before deployment, run:

```bash
python run_all_tests.py
python course_coverage_test.py
```

Deployment is only appropriate after both commands report zero failures.
