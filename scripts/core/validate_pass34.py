#!/usr/bin/env python3
"""
Validate PASS34 Results Against Ground Truth

Checks if PASS34 correctly recovered the known coefficients from synthetic data.
"""

import pandas as pd
import sys

def validate_results(results_csv, true_alpha, true_beta, true_gamma, tolerance=0.15):
    """
    Check if recovered coefficients match truth within tolerance.
    """
    df = pd.read_csv(results_csv)
    
    print("="*80)
    print("PASS34 VALIDATION RESULTS")
    print("="*80)
    print(f"\nGround Truth:")
    print(f"  b_logN (α): {true_alpha}")
    print(f"  c_logE (β): {true_beta}")
    print(f"  d_logNlogE (γ): {true_gamma}")
    print(f"\nTolerance: ±{tolerance}")
    
    # Check per-topology models with interaction
    print("\n" + "="*80)
    print("PER-TOPOLOGY MODELS (with interaction)")
    print("="*80)
    
    passed = True
    for topo in df['topology'].unique():
        sub = df[(df['topology'] == topo) & (df['model'] == 'with_cross')]
        if len(sub) == 0:
            continue
        
        row = sub.iloc[0]
        b = row['b_logN']
        c = row['c_logE']
        d = row['d_logNlogE']
        r2 = row['r2']
        
        print(f"\n{topo}:")
        print(f"  R² = {r2:.4f}")
        print(f"  b_logN recovered:   {b:7.4f}  (true: {true_alpha})")
        print(f"  c_logE recovered:   {c:7.4f}  (true: {true_beta})")
        print(f"  d_logNlogE recovered: {d:7.4f}  (true: {true_gamma})")
        
        # Check errors
        err_b = abs(b - true_alpha)
        err_c = abs(c - true_beta)
        err_d = abs(d - true_gamma)
        
        print(f"\n  Errors:")
        print(f"    |b - true_α|:  {err_b:.4f}  {'✓ PASS' if err_b < tolerance else '✗ FAIL'}")
        print(f"    |c - true_β|:  {err_c:.4f}  {'✓ PASS' if err_c < tolerance else '✗ FAIL'}")
        print(f"    |d - true_γ|:  {err_d:.4f}  {'✓ PASS' if err_d < tolerance else '✗ FAIL'}")
        
        if err_b >= tolerance or err_c >= tolerance or err_d >= tolerance:
            passed = False
    
    print("\n" + "="*80)
    if passed:
        print("✓✓✓ VALIDATION PASSED ✓✓✓")
        print("PASS34 correctly recovered all coefficients!")
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        print("Some coefficients were not recovered accurately.")
        print("\nPossible causes:")
        print("  - Insufficient signal-to-noise ratio")
        print("  - Model specification issue")
        print("  - Numerical instability")
    print("="*80)
    
    return passed


if __name__ == "__main__":
    # Validate the interaction test case
    results_file = "/home/claude/test_data_v2/pass34_results_interaction/pass34_per_topology_models.csv"
    
    # True parameters from synthetic data generation
    true_alpha = 0.5   # b_logN
    true_beta = 0.3    # c_logE  
    true_gamma = 0.2   # d_logNlogE (interaction)
    
    passed = validate_results(results_file, true_alpha, true_beta, true_gamma)
    sys.exit(0 if passed else 1)
