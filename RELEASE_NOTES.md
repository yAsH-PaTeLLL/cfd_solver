# CFD Classroom Solver v1.3.1

This is the v1.3 corrected solver package supplied for final local verification and deployment preparation.

Changes from v1.3 package:
- Added `.gitignore` for safe GitHub publishing.
- No numerical solver changes were made.
- v1.3 reported 22 app-page stand-in checks, 19 course-coverage checks, and 0 failures.

Before deployment:
1. Run `pip install -r requirements.txt`.
2. Run `python run_all_tests.py`.
3. Run `python course_coverage_test.py`.
4. Run `streamlit run app.py` and manually test the real interface.

### 1.3.2
- Adjusted the bottom-right attribution to be slightly more visible while remaining minimalist.
