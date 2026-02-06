#!/usr/bin/env python3
"""
Synthetic Data Generator for PASS47 (Topology-Controlled Spectral Regression)

Tests the critical topology × spectrum interaction that proves:
- Spectral irregularity matters differently for different topologies
- GR-like (structure-sensitive) vs Newtonian (topology-dominated) regimes

Test Cases:
1. no_interaction: gap_cv has same effect for both topologies
2. strong_interaction: gap_cv effect depends strongly on topology
3. grid_only: gap_cv only matters for grid topology (most realistic)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_pass45_join_synthetic(
    n_seeds=10,
    n_N_values=3,
    n_eps_values=5,
    gapcv_effect_rr=0.0,      # Effect of gap_cv for RR
    gapcv_effect_grid=-2.5,   # Effect of gap_cv for grid (original: -273)
    topology_offset_grid=0.5,  # Base difference for grid
    noise_std=0.1,
    seed=42
):
    """
    Generate synthetic PASS45 join data (collapse_metric + spectral metrics).
    
    Model: log(collapse) = intercept + b*logN + c*eps + d*topo + e*gap_cv + f*gap_cv*topo
    """
    rng = np.random.default_rng(seed)
    
    N_values = [2048, 4096, 8192][:n_N_values]
    eps_values = np.logspace(-4, -1, n_eps_values)
    topologies = ["rr", "grid2d_periodic"]
    dir_modes = ["fiedler"]
    probe_modes = ["add_to_dx"]
    
    rows = []
    
    for topo in topologies:
        for graph_seed in range(n_seeds):
            for N in N_values:
                # Generate gap_cv for this (topo, N, seed)
                # Grid tends to have slightly higher gap_cv (more regular → lower CV actually)
                # But let's make them overlap for testing
                if topo == "rr":
                    gap_cv_base = 0.3 + rng.normal(0, 0.1)
                else:
                    gap_cv_base = 0.35 + rng.normal(0, 0.1)
                
                gap_cv = max(0.01, gap_cv_base)  # Keep positive
                
                for eps in eps_values:
                    for dir_mode in dir_modes:
                        for probe_mode in probe_modes:
                            # Base collapse metric
                            log_collapse = (
                                -2.0 +  # Intercept
                                0.3 * np.log(N/2048) +  # Log N effect
                                0.5 * np.log(eps/0.0001) +  # Log eps effect
                                rng.normal(0, noise_std)
                            )
                            
                            # Topology effect
                            if topo == "grid2d_periodic":
                                log_collapse += topology_offset_grid
                                
                                # gap_cv interaction (CRITICAL!)
                                log_collapse += gapcv_effect_grid * gap_cv
                            else:  # rr
                                log_collapse += gapcv_effect_rr * gap_cv
                            
                            collapse_metric = np.exp(log_collapse)
                            collapse_metric = max(0.0001, collapse_metric)
                            
                            rows.append({
                                "topology": topo,
                                "N": int(N),
                                "graph_seed": int(graph_seed),
                                "probe_eps": float(eps),
                                "probe_dir_mode": dir_mode,
                                "probe_mode": probe_mode,
                                "collapse_metric": float(collapse_metric),
                                "gap_cv": float(gap_cv),
                                "logN": float(np.log(N)),
                            })
    
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="test_data_pass47", help="Output directory")
    ap.add_argument("--test_case", default="strong_interaction",
                   choices=["no_interaction", "strong_interaction", "grid_only"],
                   help="Which test case to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Set parameters based on test case
    if args.test_case == "no_interaction":
        params = {
            "gapcv_effect_rr": -1.0,
            "gapcv_effect_grid": -1.0,  # Same effect for both!
            "topology_offset_grid": 0.5,
            "noise_std": 0.15,
        }
        desc = "No interaction: gap_cv has same effect for RR and Grid"
        expected = "Interaction term not significant"
        
    elif args.test_case == "strong_interaction":
        params = {
            "gapcv_effect_rr": 0.0,      # No effect for RR
            "gapcv_effect_grid": -2.5,   # Strong effect for Grid
            "topology_offset_grid": 2.0,  # Higher baseline for grid
            "noise_std": 0.15,
        }
        desc = "Strong interaction: gap_cv only affects Grid"
        expected = "Interaction term strongly negative"
        
    else:  # grid_only (most realistic based on original results)
        params = {
            "gapcv_effect_rr": 0.1,      # Weak positive for RR
            "gapcv_effect_grid": -3.0,   # Strong negative for Grid
            "topology_offset_grid": 1.5,
            "noise_std": 0.15,
        }
        desc = "Grid-dominant: gap_cv suppresses Grid, helps RR slightly"
        expected = "Interaction term significant, topology × spectrum proven"
    
    print(f"\n{'='*70}")
    print(f"Generating synthetic PASS45 join data for PASS47: {args.test_case}")
    print(f"{'='*70}")
    print(f"\nTest case: {desc}")
    print(f"\nTrue parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Generate data
    df = generate_pass45_join_synthetic(**params, seed=args.seed)
    
    # Save
    out_path = outdir / f"synthetic_pass45_join_{args.test_case}.csv"
    df.to_csv(out_path, index=False)
    
    # Create ground truth file
    truth_path = outdir / f"synthetic_pass45_join_{args.test_case}_TRUTH.txt"
    with open(truth_path, "w") as f:
        f.write(f"SYNTHETIC DATA GROUND TRUTH\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Test case: {args.test_case}\n")
        f.write(f"Description: {desc}\n\n")
        f.write(f"True parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nExpected PASS47 results:\n")
        f.write(f"  {expected}\n")
        if args.test_case == "strong_interaction" or args.test_case == "grid_only":
            f.write(f"  → This proves hierarchical geometric structure\n")
            f.write(f"  → Spectral structure matters differently per topology\n")
        f.write(f"\nPass criterion:\n")
        f.write(f"  Interaction CI excludes zero (if interaction expected)\n")
        f.write(f"  Interaction CI includes zero (if no interaction expected)\n")
    
    print(f"\n{'='*70}")
    print(f"Generated {len(df)} synthetic data points")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {truth_path}")
    print(f"{'='*70}\n")
    
    # Quick summary stats
    print("Summary by topology:")
    print(df.groupby("topology")["collapse_metric"].describe())
    print("\nGap_cv distribution by topology:")
    print(df.groupby("topology")["gap_cv"].describe())
    
    print(f"\n✓ Ready to test with PASS47!")
    print(f"\nNext step:")
    print(f"  python pass47_topology_controlled_regression.py \\")
    print(f"    --seed_join_csv {out_path} \\")
    print(f"    --outdir {outdir}/pass47_results_{args.test_case}")


if __name__ == "__main__":
    main()
