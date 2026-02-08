#!/usr/bin/env python3
"""
PASS45 - PMIR Verification Test
Extracted from ChatGPT transcript (line 94445)
Length: 7260 characters
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def die(msg: str, code: int = 2) -> None:
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)


def safe_log(x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, floor))


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    OLS via least squares. Returns (beta, yhat, r2).
    Falls back to tiny ridge if numerically singular.
    """
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        beta = np.linalg.lstsq(X + 1e-12 * np.eye(X.shape[1]), y, rcond=None)[0]

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, r2


def bootstrap_betas(X: np.ndarray, y: np.ndarray, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y)
    p = X.shape[1]
    betas = np.zeros((reps, p), dtype=float)

    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        b, _, _ = ols_fit(X[idx], y[idx])
        betas[i] = b

    return betas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_join_csv", required=True, help="pass45_seedlevel_join.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--predictor", default="gap_cv", help="primary predictor (default gap_cv)")
    ap.add_argument(
        "--controls",
        nargs="*",
        default=["logN", "probe_eps"],
        help="control vars (default logN probe_eps)",
    )
    ap.add_argument("--boot_reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_rows_group", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.seed_join_csv)

    # Required cols (probe_dir_mode/probe_mode only needed for grouped correlations)
    need_base = ["collapse_metric", "N", "probe_eps", args.predictor]
    for c in args.controls:
        if c not in need_base:
            need_base.append(c)

    missing_base = [c for c in need_base if c not in df.columns]
    if missing_base:
        die(f"missing required columns in seed_join_csv: {missing_base}")

    # Filter to finite rows
    d = df.copy()
    d = d[np.isfinite(d["collapse_metric"].astype(float))]
    d = d[np.isfinite(d[args.predictor].astype(float))].copy()

    # Ensure logN exists if requested
    if "logN" in args.controls and "logN" not in d.columns:
        d["logN"] = safe_log(d["N"].to_numpy(dtype=float))

    # y = log(collapse_metric + eps) to stabilize scale
    y_raw = d["collapse_metric"].to_numpy(dtype=float)
    y = safe_log(y_raw + 1e-30)

    def build_X(cols):
        X_parts = [np.ones(len(d), dtype=float)]
        names = ["c1"]
        for c in cols:
            X_parts.append(d[c].to_numpy(dtype=float))
            names.append(c)
        return np.column_stack(X_parts), names

    # Models
    models = []
    X0, n0 = build_X(args.controls)
    models.append(("M0_logY~1+controls", X0, n0))

    X1, n1 = build_X(args.controls + [args.predictor])
    models.append((f"M1_logY~1+controls+{args.predictor}", X1, n1))

    X2, n2 = build_X([args.predictor])
    models.append((f"M2_logY~1+{args.predictor}", X2, n2))

    rows = []
    txt_lines = []
    txt_lines.append("PASS45 — seed-level spectral regression\n\n")
    txt_lines.append(f"seed_join_csv: {args.seed_join_csv}\n")
    txt_lines.append(f"predictor: {args.predictor}\n")
    txt_lines.append(f"controls: {args.controls}\n")
    txt_lines.append(f"boot_reps: {args.boot_reps} seed: {args.seed}\n")
    txt_lines.append(f"rows used (after dropna): {len(d)}\n\n")

    for name, X, colnames in models:
        beta, _, r2 = ols_fit(X, y)
        betas = bootstrap_betas(X, y, reps=args.boot_reps, seed=args.seed)

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
        txt_lines.append(f"  r2: {r2:.6f}\n")
        for j, cn in enumerate(colnames):
            txt_lines.append(
                f"  {cn}: beta={beta[j]:.6g}  med={med[j]:.6g}  CI=[{lo[j]:.6g},{hi[j]:.6g}]\n"
            )
        txt_lines.append("\n")

    out_csv = os.path.join(args.outdir, "pass45_seedlevel_regression_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Grouped correlations (robustness by probe group)
    out_grp = os.path.join(args.outdir, "pass45_seedlevel_group_corr.csv")

    grp_cols = ["probe_dir_mode", "probe_mode", "probe_eps"]
    has_group_cols = all(c in d.columns for c in grp_cols)

    grp_rows = []
    if has_group_cols:
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

    dfg = pd.DataFrame(grp_rows)
    if len(dfg) > 0:
        dfg = dfg.sort_values(["probe_dir_mode", "probe_mode", "probe_eps"])
        dfg.to_csv(out_grp, index=False)
        txt_lines.append("Grouped correlations (logY vs predictor)\n")
        txt_lines.append(f"  group_cols: {grp_cols}\n")
        txt_lines.append(f"  min_rows_group: {args.min_rows_group}\n")
        txt_lines.append(f"  wrote: {out_grp}\n")
    else:
        # Still write an empty file so downstream scripts don't crash
        pd.DataFrame(
            columns=["probe_dir_mode", "probe_mode", "probe_eps", "n_rows", "corr", "x_mean", "y_mean"]
        ).to_csv(out_grp, index=False)

        txt_lines.append("Grouped correlations (logY vs predictor)\n")
        if not has_group_cols:
            txt_lines.append(f"  skipped: missing group cols {grp_cols}\n")
        else:
            txt_lines.append("  skipped: no groups met min_rows_group (or zero variance)\n")
        txt_lines.append(f"  wrote empty: {out_grp}\n")

    out_txt = os.path.join(args.outdir, "pass45_seedlevel_regression_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("".join(txt_lines))

    print("[PASS45] complete")
    print(f"[PASS45] wrote {out_csv}")
    print(f"[PASS45] wrote {out_grp}")
    print(f"[PASS45] wrote {out_txt}")


if __name__ == "__main__":
    main()