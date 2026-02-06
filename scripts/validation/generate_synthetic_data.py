#!/usr/bin/env python3
"""
Synthetic Data Generator for PASS34 Validation

Generates test data with known scaling laws to verify PASS34 correctly
recovers the coefficients and detects interactions.

Test Cases:
1. No interaction: Δ ~ N^0.5 × ε^0.3
2. With interaction: Δ ~ N^0.5 × ε^0.3 × (N×ε)^0.2
3. Topology dependence: Different slopes for RR vs grid
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_pass33_data(
    n_points=100,
    alpha=0.5,      # log N coefficient
    beta=0.3,       # log ε coefficient  
    gamma=0.0,      # log N × log ε interaction
    noise_std=0.1,
    topo_offset=0.5,  # Additional offset for grid vs RR
    topo_scale_diff=0.2,  # Different scaling for grid
    seed=42
):
    """
    Generate synthetic data mimicking PASS33 output with known scaling law.
    
    True model:
        log(Δ) = a + α*log(N) + β*log(ε) + γ*log(N)*log(ε) + noise
        
    For grid topology, add:
        + topo_offset + topo_scale_diff*log(N)
    """
    rng = np.random.default_rng(seed)
    
    # Generate parameter combinations
    N_values = [2048, 4096, 8192]
    eps_values = np.logspace(-4, -1, 15)  # ε from 0.0001 to 0.1
    topologies = ["rr", "grid2d_periodic"]
    
    rows = []
    for topo in topologies:
        for N in N_values:
            for eps in eps_values:
                # Base scaling law
                log_N = np.log(N)
                log_eps = np.log(eps)
                
                # True model: LOG-LINEAR (what PASS34 actually fits!)
                # Δ = a + b*log(N) + c*log(ε) + d*log(N)*log(ε)
                
                intercept = 0.5  # Baseline value
                delta_true = (
                    intercept +
                    alpha * log_N + 
                    beta * log_eps + 
                    gamma * log_N * log_eps
                )
                
                # Topology effect (grid is "heavier")
                if topo == "grid2d_periodic":
                    delta_true += topo_offset + topo_scale_diff * log_N
                
                # Add noise
                delta_noisy = delta_true + rng.normal(0, noise_std)
                delta = delta_noisy  # No exp() - we're in linear space!
                
                # Synthetic permutation test p-value (would be significant)
                # Make more significant when true effect is larger
                p_perm = 0.001 if abs(delta) > 0.1 else 0.05
                
                rows.append({
                    "topology": topo,
                    "N": int(N),
                    "probe_eps": float(eps),
                    "delta_mean_a_minus_b": float(delta),
                    "p_perm": float(p_perm),
                    "probe_dir_mode": "fiedler",  # Could vary
                    "probe_mode": "add_to_dx",
                    # Ground truth for validation
                    "delta_true": float(delta_true),
                    "delta_noisy": float(delta_noisy),
                })
    
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="test_data", help="Output directory")
    ap.add_argument("--test_case", default="interaction", 
                   choices=["no_interaction", "interaction", "topology_only"],
                   help="Which test case to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Set parameters based on test case
    if args.test_case == "no_interaction":
        params = {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.0,  # NO INTERACTION
            "topo_offset": 0.0,
            "topo_scale_diff": 0.0,
            "noise_std": 0.1,
        }
        desc = "No interaction: Δ ~ N^0.5 × ε^0.3"
        
    elif args.test_case == "interaction":
        params = {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2,  # INTERACTION PRESENT
            "topo_offset": 0.0,
            "topo_scale_diff": 0.0,
            "noise_std": 0.1,
        }
        desc = "With interaction: Δ ~ N^0.5 × ε^0.3 × (N×ε)^0.2"
        
    else:  # topology_only
        params = {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.0,
            "topo_offset": 0.5,      # Grid higher baseline
            "topo_scale_diff": 0.2,  # Grid scales differently with N
            "noise_std": 0.1,
        }
        desc = "Topology dependence: Grid vs RR with different scaling"
    
    print(f"\n{'='*70}")
    print(f"Generating synthetic PASS33 data: {args.test_case}")
    print(f"{'='*70}")
    print(f"\nTest case: {desc}")
    print(f"\nTrue parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Generate data
    df = generate_synthetic_pass33_data(**params, seed=args.seed)
    
    # Save
    out_path = outdir / f"synthetic_pass33_{args.test_case}.csv"
    df.to_csv(out_path, index=False)
    
    # Create ground truth file
    truth_path = outdir / f"synthetic_pass33_{args.test_case}_TRUTH.txt"
    with open(truth_path, "w") as f:
        f.write(f"SYNTHETIC DATA GROUND TRUTH\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Test case: {args.test_case}\n")
        f.write(f"Description: {desc}\n\n")
        f.write(f"True parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nExpected PASS34 results:\n")
        f.write(f"  b_logN should recover: {params['alpha']}\n")
        f.write(f"  c_logE should recover: {params['beta']}\n")
        f.write(f"  g_logNlogE should recover: {params['gamma']}\n")
        if args.test_case == "topology_only":
            f.write(f"  Topology dummy interactions should be significant\n")
        f.write(f"\nPass criterion:\n")
        f.write(f"  |recovered - true| < 0.15 (allowing for noise)\n")
    
    print(f"\n{'='*70}")
    print(f"Generated {len(df)} synthetic data points")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {truth_path}")
    print(f"{'='*70}\n")
    
    # Quick summary stats
    print("Summary by topology:")
    print(df.groupby("topology")["delta_mean_a_minus_b"].describe())
    print("\nSummary by N:")
    print(df.groupby("N")["delta_mean_a_minus_b"].describe())
    
    print(f"\n✓ Ready to test with PASS34!")
    print(f"\nNext step:")
    print(f"  python pass34_scaling_regression_auc.py \\")
    print(f"    --in_csv {out_path} \\")
    print(f"    --outdir {outdir}/pass34_results_{args.test_case} \\")
    print(f"    --topo_ref rr")


if __name__ == "__main__":
    main()
