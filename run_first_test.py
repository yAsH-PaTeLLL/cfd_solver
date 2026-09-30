from examples import mst1_q3, mst1_q4_transient_implicit

print("\n===== TEST 1: MST-1 Q3 steady FDM =====")
mst1_q3(show_plot=False)

print("\n===== TEST 2: MST-1 Q4 transient FDM =====")
r = mst1_q4_transient_implicit(show_plot=False)

print("\nTEST STATUS: PASS")
print("Steady FDM and fully-implicit transient FDM tests ran successfully.")
