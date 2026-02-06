#!/usr/bin/env python3
"""
PASS45 - Seed-Level Spectral Join

Joins collapse metrics (PASS33 seed-level) with spectral metrics (PASS42)
on (topology, N, graph_seed) to enable seed-level regression analysis.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def die(msg: str, code: int = 2):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_cols(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"missing column(s) in {label}: {missing}")


def safe_log(x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, floor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse_csv", required=True, help="pass33_by_graph_eps_dir_probe.csv")
    ap.add_argument("--spectral_csv", required=True, help="pass42_eigs_band_metrics.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--collapse_metric", default="score_mean", help="metric column in collapse csv")
    ap.add_argument("--seed_col", default="graph_seed", help="seed column name in collapse csv")
    ap.add_argument("--use_abs", type=int, default=1, help="absolute value for collapse_metric")
    ap.add_argument("--keep_all_spectral", type=int, default=0, help="keep all spectral cols")
    ap.add_argument("--drop_quarantine", type=int, default=1, help="drop quarantine rows")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Load collapse data
    c = pd.read_csv(args.collapse_csv)
    ensure_cols(
        c,
        ["topology", "N", args.seed_col, "probe_eps", "probe_dir_mode", "probe_mode", args.collapse_metric],
        "collapse_csv",
    )

    c = c.copy()
    c["N"] = c["N"].astype(int)
    c[args.seed_col] = c[args.seed_col].astype(int)
    c["collapse_metric"] = c[args.collapse_metric].astype(float)
    
    if args.use_abs == 1:
        c["collapse_metric"] = c["collapse_metric"].abs()

    c["logN"] = safe_log(c["N"].to_numpy(dtype=float))

    # Load spectral data
    s = pd.read_csv(args.spectral_csv)
    ensure_cols(s, ["topology", "N", "seed", "gap_cv", "gap_mean", "gap_std"], "spectral_csv")

    s = s.copy()
    s["N"] = s["N"].astype(int)
    s["seed"] = s["seed"].astype(int)

    if args.drop_quarantine == 1 and "file" in s.columns:
        s = s[~s["file"].astype(str).str.contains("_quarantine", na=False)].copy()

    # Rename seed -> graph_seed for join
    s = s.rename(columns={"seed": args.seed_col})

    # Compute safe ratio if available
    if "gap_max" in s.columns and "gap_min" in s.columns:
        gap_min_floor = 1e-14
        s["gap_ratio_safe"] = s["gap_max"].astype(float) / np.maximum(s["gap_min"].astype(float), gap_min_floor)
    elif "gap_max_over_min" in s.columns:
        s["gap_ratio_safe"] = s["gap_max_over_min"].astype(float)

    # Keep only needed spectral columns
    base_spec_cols = [
        "topology",
        "N",
        args.seed_col,
        "gap_cv",
        "gap_mean",
        "gap_std",
    ]
    if "gap_ratio_safe" in s.columns:
        base_spec_cols.append("gap_ratio_safe")
    
    if args.keep_all_spectral == 1:
        spec_cols = s.columns.tolist()
    else:
        spec_cols = [col for col in base_spec_cols if col in s.columns]

    s = s[spec_cols].copy()

    # Deduplicate spectral
    s = s.drop_duplicates(subset=["topology", "N", args.seed_col], keep="first")

    # Join
    j = c.merge(s, on=["topology", "N", args.seed_col], how="left", validate="many_to_one")

    # Join diagnostics
    n_total = len(j)
    n_matched = int(j["gap_cv"].notna().sum())
    n_missing = n_total - n_matched

    # Write outputs
    out_join = os.path.join(outdir, "pass45_seedlevel_join.csv")
    j.to_csv(out_join, index=False)

    # Summary
    out_sum = os.path.join(outdir, "pass45_seedlevel_join_summary.txt")
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("PASS45 — seed-level spectral join\n")
        f.write(f"collapse_csv: {args.collapse_csv}\n")
        f.write(f"spectral_csv: {args.spectral_csv}\n")
        f.write(f"out_join_csv: {out_join}\n")
        f.write("\n")
        f.write(f"rows_total: {n_total}\n")
        f.write(f"rows_with_gap_cv: {n_matched}\n")
        f.write(f"rows_missing_gap_cv: {n_missing}\n")
        f.write("\n")
        f.write("counts by topology:\n")
        f.write(j.groupby("topology").size().to_string())
        f.write("\n\n")
        f.write("gap_cv availability by topology:\n")
        f.write(j.groupby("topology")["gap_cv"].apply(lambda x: int(x.notna().sum())).to_string())
        f.write("\n\n")
        f.write("N unique (collapse): " + str(c["N"].nunique()) + "\n")
        f.write("N unique (spectral): " + str(s["N"].nunique()) + "\n")
        f.write("N unique (joined): " + str(j["N"].nunique()) + "\n")
        f.write("\n")
        if n_missing > 0:
            miss = j[j["gap_cv"].isna()][["topology", "N", args.seed_col]].drop_duplicates().head(25)
            f.write("sample missing spectral keys (topology,N,seed):\n")
            f.write(miss.to_string(index=False))
            f.write("\n")

    print("[PASS45] complete")
    print(f"[PASS45] wrote {out_join}")
    print(f"[PASS45] wrote {out_sum}")


if __name__ == "__main__":
    main()
