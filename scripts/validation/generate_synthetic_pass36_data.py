#!/usr/bin/env python3
"""
Synthetic Data Generator for PASS36 (Fixed-Point Collapse Test)

Generates PASS33-like summary data with known fixed-point behavior.

Test Cases:
1. perfect_fixedpoint: Ratio remains constant across N (CV → 0)
2. scale_dependent: Ratio changes systematically with N (high CV)
3. mixed: Some parameter combinations invariant, others not
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_pass33_summary_synthetic(
    n_N_values=3,
    n_eps_values=5,
    n_dir_modes=2,
    fixed_point_ratio=1.5,  # Grid/RR ratio at fixed point
    scale_dependence=0.0,   # How much ratio changes with N
    noise_std=0.05,
    seed=42
):
    """
    Generate synthetic PASS33 summary data.
    
    For fixed-point behavior: ratio stays constant across N
    For scale-dependent: ratio = base_ratio * (N / N_min)^scale_dependence
    """
    rng = np.random.default_rng(seed)
    
    N_values = [2048, 4096, 8192][:n_N_values]
    eps_values = np.logspace(-4, -1, n_eps_values)
    dir_modes = ["fiedler", "random"][:n_dir_modes]
    probe_modes = ["add_to_dx"]  # Can expand
    topologies = ["rr", "grid2d_periodic"]
    
    rows = []
    
    for dir_mode in dir_modes:
        for probe_mode in probe_modes:
            for eps in eps_values:
                # Base score for RR at smallest N
                base_score_rr = 0.5 + 0.3 * np.log10(eps) + rng.normal(0, 0.1)
                base_score_rr = max(0.01, base_score_rr)  # Keep positive
                
                for N in N_values:
                    # RR scaling
                    score_rr = base_score_rr * (N / N_values[0])**0.3
                    score_rr += rng.normal(0, noise_std)
                    score_rr = max(0.001, score_rr)
                    
                    # Grid scaling - depends on whether this is a fixed point
                    scale_factor = (N / N_values[0])**scale_dependence
                    ratio = fixed_point_ratio * scale_factor
                    
                    score_grid = score_rr * ratio
                    score_grid += rng.normal(0, noise_std)
                    score_grid = max(0.001, score_grid)
                    
                    # Add RR row
                    rows.append({
                        "probe_dir_mode": dir_mode,
                        "probe_mode": probe_mode,
                        "topology": "rr",
                        "N": int(N),
                        "probe_eps": float(eps),
                        "mean_score": float(score_rr),
                    })
                    
                    # Add Grid row
                    rows.append({
                        "probe_dir_mode": dir_mode,
                        "probe_mode": probe_mode,
                        "topology": "grid2d_periodic",
                        "N": int(N),
                        "probe_eps": float(eps),
                        "mean_score": float(score_grid),
                    })
    
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="test_data_pass36", help="Output directory")
    ap.add_argument("--test_case", default="perfect_fixedpoint",
                   choices=["perfect_fixedpoint", "scale_dependent", "mixed"],
                   help="Which test case to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Set parameters based on test case
    if args.test_case == "perfect_fixedpoint":
        params = {
            "fixed_point_ratio": 1.5,
            "scale_dependence": 0.0,  # NO scale dependence → fixed point
            "noise_std": 0.03,
        }
        desc = "Perfect fixed point: Grid/RR ratio constant across N"
        expected_cv = "< 0.10"
        
    elif args.test_case == "scale_dependent":
        params = {
            "fixed_point_ratio": 1.5,
            "scale_dependence": 0.3,  # Strong scale dependence
            "noise_std": 0.03,
        }
        desc = "Scale dependent: Grid/RR ratio changes with N"
        expected_cv = "> 0.20"
        
    else:  # mixed
        # This would require more complex generation
        # For now, use moderate scale dependence
        params = {
            "fixed_point_ratio": 1.5,
            "scale_dependence": 0.15,
            "noise_std": 0.05,
        }
        desc = "Mixed: Some invariance, some scale-dependence"
        expected_cv = "0.10 - 0.20"
    
    print(f"\n{'='*70}")
    print(f"Generating synthetic PASS33 summary for PASS36: {args.test_case}")
    print(f"{'='*70}")
    print(f"\nTest case: {desc}")
    print(f"\nTrue parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Generate data
    df = generate_pass33_summary_synthetic(**params, seed=args.seed)
    
    # Save
    out_path = outdir / f"synthetic_pass33_summary_{args.test_case}.csv"
    df.to_csv(out_path, index=False)
    
    # Create ground truth file
    truth_path = outdir / f"synthetic_pass33_summary_{args.test_case}_TRUTH.txt"
    with open(truth_path, "w") as f:
        f.write(f"SYNTHETIC DATA GROUND TRUTH\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Test case: {args.test_case}\n")
        f.write(f"Description: {desc}\n\n")
        f.write(f"True parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nExpected PASS36 results:\n")
        f.write(f"  ratio_cv_across_N: {expected_cv}\n")
        if args.test_case == "perfect_fixedpoint":
            f.write(f"  → Should show fixed-point behavior\n")
        elif args.test_case == "scale_dependent":
            f.write(f"  → Should show strong scale-dependence\n")
        f.write(f"\nPass criterion:\n")
        f.write(f"  CV matches expected range\n")
    
    print(f"\n{'='*70}")
    print(f"Generated {len(df)} synthetic data points")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {truth_path}")
    print(f"{'='*70}\n")
    
    # Quick summary stats
    print("Summary by topology:")
    print(df.groupby("topology")["mean_score"].describe())
    
    # Show example grid/rr ratios
    print("\nExample Grid/RR ratios (should be ~1.5 for all N if fixed-point):")
    for N in df["N"].unique():
        sub = df[(df["N"] == N) & (df["probe_dir_mode"] == "fiedler") & (df["probe_eps"].round(6) == df["probe_eps"].unique()[2])]
        if len(sub) == 2:
            rr_score = sub[sub["topology"] == "rr"]["mean_score"].values[0]
            grid_score = sub[sub["topology"] == "grid2d_periodic"]["mean_score"].values[0]
            ratio = grid_score / rr_score if rr_score > 0 else np.nan
            print(f"  N={N}: ratio = {ratio:.3f}")
    
    print(f"\n✓ Ready to test with PASS36!")
    print(f"\nNext step:")
    print(f"  python pass36_fixed_point_collapse.py \\")
    print(f"    --in_csv {out_path} \\")
    print(f"    --outdir {outdir}/pass36_results_{args.test_case}")


if __name__ == "__main__":
    main()
