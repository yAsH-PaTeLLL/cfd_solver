# Final Local Acceptance Check

1. Create/activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run numerical tests:
   `python run_all_tests.py`
4. Run course coverage:
   `python course_coverage_test.py`
5. Launch the UI:
   `streamlit run app.py`
6. In the browser, test Home, Course Problems, Custom FDM, Custom FVM,
   Convection-Diffusion, PDE + Method Guide, and Verification.
7. Test at least one known course problem and one new custom problem.
8. Only after the local UI test passes, push this folder to GitHub and deploy.
