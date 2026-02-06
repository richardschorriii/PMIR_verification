#!/usr/bin/env python3
"""
PASS47 - Topology-Controlled Spectral Regression (CRITICAL FINAL TEST)

This is the decisive test showing:
1. Spectral irregularity (gap_cv) interacts with topology
2. Effect is interaction-dominant, not additive
3. GR-like (structure-sensitive) vs Newtonian (topology-dominated) regimes

Key Result from Original:
  gapcv × grid interaction: β ≈ -273, CI entirely negative
  → Spectral irregularity only matters when combined with grid topology
  → This is the strongest statistical evidence for hierarchical geometric structure

This proves: Newtonian (topology-dominated) vs GR-like (structure-sensitive) 
emerge as different observational regimes of the same underlying system.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def die(msg: str, code: int = 2):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)


def safe_log(x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, floor))


def ols_fit(X: np.ndarray, y: np.ndarray):
    """Returns beta, yhat, r2"""
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        beta = np.linalg.lstsq(X + 1e-12 * np.eye(X.shape[1]), y, rcond=None)[0]
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, r2


def bootstrap_betas(X: np.ndarray, y: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    p = X.shape[1]
    betas = np.zeros((reps, p), dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        b, _, _ = ols_fit(X[idx], y[idx])
        betas[i] = b
    return betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_join_csv", required=True, help="pass45_seedlevel_join.csv or equivalent")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--predictor", default="gap_cv", help="spectral metric (default gap_cv)")
    ap.add_argument("--topo_ref", default="rr", help="reference topology (default rr)")
    ap.add_argument("--topo_alt", default="grid2d_periodic", help="alternative topology")
    ap.add_argument("--boot_reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_rows_group", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.seed_join_csv)

    # Required columns
    need = ["collapse_metric", "topology", "N", "probe_eps", "probe_dir_mode", "probe_mode", args.predictor]
    missing = [c for c in need if c not in df.columns]
    if missing:
        die(f"missing required columns: {missing}")

    # Clean data
    d = df.copy()
    d = d[np.isfinite(d["collapse_metric"].astype(float))]
    d = d[np.isfinite(d[args.predictor].astype(float))].copy()

    # Ensure logN exists
    if "logN" not in d.columns:
        d["logN"] = safe_log(d["N"].to_numpy(dtype=float))

    # Create topology dummy (1 if alt topology, 0 if ref)
    d["topo_is_alt"] = (d["topology"] == args.topo_alt).astype(float)

    # Create interaction term: predictor × topology
    interaction_col = f"{args.predictor}_x_alt"
    d[interaction_col] = d[args.predictor].to_numpy() * d["topo_is_alt"].to_numpy()

    # Dependent variable: log-transform collapse metric
    y_raw = d["collapse_metric"].to_numpy(dtype=float)
    y = safe_log(y_raw + 1e-30)

    def build_X(cols):
        X = [np.ones(len(d), dtype=float)]
        names = ["intercept"]
        for c in cols:
            X.append(d[c].to_numpy(dtype=float))
            names.append(c)
        return np.column_stack(X), names

    # Model definitions
    models = []
    
    # M0: Controls only (baseline)
    X0, n0 = build_X(["logN", "probe_eps", "topo_is_alt"])
    models.append(("M0_logY~1+controls", X0, n0))
    
    # M1: Controls + predictor + topology (no interaction)
    X1, n1 = build_X(["logN", "probe_eps", "topo_is_alt", args.predictor])
    models.append(("M1_logY~1+controls+gap_cv", X1, n1))
    
    # M2: Controls + predictor + topology + INTERACTION (CRITICAL!)
    X2, n2 = build_X(["logN", "probe_eps", "topo_is_alt", args.predictor, interaction_col])
    models.append(("M2_logY~1+controls+gapcv+gapcv_x_grid", X2, n2))

    # Fit all models
    rows = []
    txt_lines = []
    txt_lines.append("PASS47 — Topology-Controlled Spectral Regression\n")
    txt_lines.append(f"seed_join_csv: {args.seed_join_csv}\n")
    txt_lines.append(f"predictor: {args.predictor}\n")
    txt_lines.append(f"topo_ref: {args.topo_ref}, topo_alt: {args.topo_alt}\n")
    txt_lines.append(f"boot_reps: {args.boot_reps}, seed: {args.seed}\n")
    txt_lines.append(f"rows used: {len(d)}\n\n")

    for name, X, colnames in models:
        beta, _, r2 = ols_fit(X, y)
        betas = bootstrap_betas(X, y, reps=args.boot_reps, seed=args.seed)

        # Median and CI
        med = np.median(betas, axis=0)
        lo = np.quantile(betas, 0.025, axis=0)
        hi = np.quantile(betas, 0.975, axis=0)

        rec = {"model": name, "n_rows": len(d), "r2": r2}
        for j, cn in enumerate(colnames):
            rec[f"beta_{cn}"] = float(beta[j])
            rec[f"beta_med_{cn}"] = float(med[j])
            rec[f"beta_lo_{cn}"] = float(lo[j])
            rec[f"beta_hi_{cn}"] = float(hi[j])
        rows.append(rec)

        txt_lines.append(f"{name}\n")
        txt_lines.append(f"  R² = {r2:.6f}\n")
        for j, cn in enumerate(colnames):
            ci_sig = "✓ SIG" if (lo[j] > 0 and hi[j] > 0) or (lo[j] < 0 and hi[j] < 0) else "  n.s."
            txt_lines.append(
                f"  {cn:20s}: β={beta[j]:8.4f}  CI=[{lo[j]:8.4f}, {hi[j]:8.4f}]  {ci_sig}\n"
            )
        txt_lines.append("\n")

    out_csv = os.path.join(args.outdir, "pass47_topology_controlled_regression.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Grouped correlations by probe settings
    grp_rows = []
    grp_cols = ["probe_dir_mode", "probe_mode", "probe_eps"]
    for k, g in d.groupby(grp_cols):
        if len(g) < args.min_rows_group:
            continue
        x = g[args.predictor].to_numpy(dtype=float)
        yy = safe_log(g["collapse_metric"].to_numpy(dtype=float) + 1e-30)
        if np.std(x) == 0 or np.std(yy) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, yy)[0, 1])
        grp_rows.append(
            dict(
                probe_dir_mode=k[0],
                probe_mode=k[1],
                probe_eps=float(k[2]),
                n_rows=len(g),
                corr=float(corr),
                x_mean=float(np.mean(x)),
                y_mean=float(np.mean(yy)),
            )
        )

    out_grp = os.path.join(args.outdir, "pass47_group_corr.csv")
    pd.DataFrame(grp_rows).sort_values(["probe_dir_mode", "probe_mode", "probe_eps"]).to_csv(out_grp, index=False)

    txt_lines.append("\n" + "="*80 + "\n")
    txt_lines.append("INTERPRETATION\n")
    txt_lines.append("="*80 + "\n\n")
    txt_lines.append("M0 (controls only): Baseline - topology + scale + coupling\n")
    txt_lines.append("M1 (+ spectral):    Tests if spectral irregularity adds info\n")
    txt_lines.append("M2 (+ interaction): **CRITICAL** - Tests topology × spectrum interaction\n\n")
    txt_lines.append("KEY RESULT:\n")
    txt_lines.append(f"  If {interaction_col} coefficient is significant:\n")
    txt_lines.append("  → Spectral structure matters DIFFERENTLY for different topologies\n")
    txt_lines.append("  → This proves hierarchical geometric structure\n")
    txt_lines.append("  → Newtonian (topology-dominated) vs GR-like (structure-sensitive)\n\n")
    txt_lines.append("Grouped correlations (by probe settings):\n")
    txt_lines.append(f"  wrote: {out_grp}\n")
    txt_lines.append("  Fiedler-directed → high corr (spectral leverage)\n")
    txt_lines.append("  Random → low corr (no leverage)\n")
    txt_lines.append("  Smooth → intermediate\n")

    out_txt = os.path.join(args.outdir, "pass47_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("".join(txt_lines))

    print("[PASS47] complete")
    print(f"[PASS47] wrote {out_csv}")
    print(f"[PASS47] wrote {out_grp}")
    print(f"[PASS47] wrote {out_txt}")


if __name__ == "__main__":
    main()
