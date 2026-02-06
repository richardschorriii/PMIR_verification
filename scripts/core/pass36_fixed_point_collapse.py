#!/usr/bin/env python3
"""
PASS36 - Fixed-Point Collapse Test

Tests whether the topology-dependent effects show "fixed-point-like" behavior:
- Does the rr ↔ grid contrast remain invariant across different N?
- Low CV across N suggests universal scaling behavior
- High CV suggests strong finite-size effects

This tests whether PMIR exhibits attractor-like behavior where certain
parameter combinations converge to stable values regardless of scale.

Key Metric: CV (coefficient of variation) across N
- Low CV (< 0.15) → Fixed-point-like (scale-invariant ratio)
- High CV → Strong scale-dependence
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _parse_agg_over(s: str):
    """Parse comma-separated aggregation columns"""
    if not s.strip():
        return []
    cols = [c.strip() for c in s.split(",") if c.strip()]
    # Allow shorthand
    mapping = {"dir_mode": "probe_dir_mode"}
    return [mapping.get(c, c) for c in cols]


def fit_loglinear(X, y):
    """OLS with intercept; returns beta, yhat, r2"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return beta, yhat, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="PASS33 summary with mean_score")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topo_ref", default="rr", help="Reference topology")
    ap.add_argument("--topo_alt", default="grid2d_periodic", help="Alternative topology")
    ap.add_argument("--use_abs", type=int, default=1, help="Use abs(mean_score)")
    ap.add_argument("--agg_over", default="probe_dir_mode,probe_mode,probe_eps")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--boot_reps", type=int, default=5000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.in_csv)
    
    # Required columns from PASS33 summary
    need = ["probe_dir_mode", "probe_mode", "topology", "N", "probe_eps", "mean_score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing required columns: {miss}")

    # Numeric hygiene
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["probe_eps"] = pd.to_numeric(df["probe_eps"], errors="coerce")
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df = df.dropna(subset=["probe_dir_mode", "probe_mode", "topology", "N", "probe_eps", "mean_score"])

    if args.use_abs:
        df["score_use"] = df["mean_score"].abs()
    else:
        df["score_use"] = df["mean_score"]

    # Pair rr vs grid for each (dir, mode, N, eps)
    key = ["probe_dir_mode", "probe_mode", "N", "probe_eps"]
    a = df[df["topology"] == args.topo_ref][key + ["score_use"]].rename(columns={"score_use": "score_ref"})
    b = df[df["topology"] == args.topo_alt][key + ["score_use"]].rename(columns={"score_use": "score_alt"})
    P = a.merge(b, on=key, how="inner")
    
    if len(P) == 0:
        raise SystemExit("[ERR] no paired rows found (check topo names).")

    P["delta_alt_minus_ref"] = P["score_alt"] - P["score_ref"]
    P["ratio_alt_over_ref"] = P["score_alt"] / np.where(P["score_ref"] == 0, np.nan, P["score_ref"])

    p_pairs = outdir / "pass36_pairs_topoN_eps.csv"
    P.to_csv(p_pairs, index=False)

    # Invariance diagnostics across N within each (dir, mode, eps)
    agg_over = _parse_agg_over(args.agg_over)
    for c in agg_over:
        if c not in P.columns:
            raise SystemExit(f"[ERR] agg_over column not found: {c}")

    rows = []
    for gk, sub in P.groupby(agg_over):
        sub = sub.sort_values("N")
        Ns = sub["N"].to_numpy()
        ratio = sub["ratio_alt_over_ref"].to_numpy()
        delta = sub["delta_alt_minus_ref"].to_numpy()

        def cv(x):
            """Coefficient of variation"""
            x = x[np.isfinite(x)]
            if len(x) < 2:
                return np.nan
            m = float(np.mean(x))
            s = float(np.std(x, ddof=1))
            return np.nan if m == 0 else s / abs(m)

        rows.append({
            **{agg_over[i]: gk[i] if isinstance(gk, tuple) else gk for i in range(len(agg_over))},
            "nN": int(sub["N"].nunique()),
            "ratio_mean": float(np.nanmean(ratio)),
            "ratio_cv_across_N": float(cv(ratio)),
            "delta_mean": float(np.nanmean(delta)),
            "delta_cv_across_N": float(cv(delta)),
        })

    INV = pd.DataFrame(rows).sort_values(["probe_eps", "probe_dir_mode", "probe_mode"])
    p_inv = outdir / "pass36_invariance_by_dir_mode_eps.csv"
    INV.to_csv(p_inv, index=False)

    # Collapse fit: score ~ N^b * eps^c
    # Fit log(score) = a + b*logN + c*logE + d*logN*logE
    rng = np.random.default_rng(args.seed)

    def boot_fit(subdf, reps):
        x1 = np.log(subdf["N"].to_numpy())
        x2 = np.log(subdf["probe_eps"].to_numpy())
        y = np.log(np.clip(subdf["score_use"].to_numpy(), 1e-300, np.inf))
        X = np.column_stack([x1, x2, x1 * x2])
        beta, _, r2 = fit_loglinear(X, y)
        
        betas = []
        for _ in range(reps):
            idx = rng.integers(0, len(subdf), size=len(subdf))
            b2, _, _ = fit_loglinear(X[idx], y[idx])
            betas.append(b2)
        B = np.vstack(betas)
        lo = np.quantile(B, 0.025, axis=0)
        hi = np.quantile(B, 0.975, axis=0)
        return beta, lo, hi, r2

    fit_rows = []
    for topo in [args.topo_ref, args.topo_alt]:
        sub = df[df["topology"] == topo].copy()
        sub = sub.dropna(subset=["N", "probe_eps", "score_use"])
        # Keep only positive values for logs
        sub = sub[(sub["probe_eps"] > 0) & (sub["score_use"] > 0)]
        
        if len(sub) < 10:
            fit_rows.append({"topology": topo, "status": "too_few", "n": int(len(sub))})
            continue
            
        beta, lo, hi, r2 = boot_fit(sub, args.boot_reps)
        fit_rows.append({
            "topology": topo, "status": "ok", "n": int(len(sub)),
            "a": float(beta[0]), 
            "b_logN": float(beta[1]), 
            "c_logE": float(beta[2]), 
            "d_logNlogE": float(beta[3]),
            "a_lo": float(lo[0]), 
            "b_lo": float(lo[1]), 
            "c_lo": float(lo[2]), 
            "d_lo": float(lo[3]),
            "a_hi": float(hi[0]), 
            "b_hi": float(hi[1]), 
            "c_hi": float(hi[2]), 
            "d_hi": float(hi[3]),
            "r2": float(r2),
        })

    FIT = pd.DataFrame(fit_rows)
    p_fit = outdir / "pass36_collapse_fit.csv"
    FIT.to_csv(p_fit, index=False)

    # Text summary
    txt = outdir / "pass36_fixed_point_summary.txt"
    lines = []
    lines.append("PASS36 — Fixed-Point Collapse Test")
    lines.append(f"in: {args.in_csv}")
    lines.append(f"topo_ref: {args.topo_ref}, topo_alt: {args.topo_alt}, use_abs={args.use_abs}")
    lines.append(f"paired rows: {len(P)}")
    lines.append("")
    lines.append("WROTE:")
    lines.append(f"  {p_pairs}")
    lines.append(f"  {p_inv}")
    lines.append(f"  {p_fit}")
    lines.append("")
    lines.append("INVARIANCE ANALYSIS:")
    lines.append("  CV < 0.15 → Fixed-point-like (scale-invariant)")
    lines.append("  CV > 0.30 → Strong scale-dependence")
    lines.append("")
    lines.append("Top 10 most invariant (lowest ratio_cv_across_N):")
    if len(INV):
        best = INV.sort_values("ratio_cv_across_N").head(10)
        lines.append(best.to_string(index=False))
    else:
        lines.append("  [none]")
    lines.append("")
    lines.append("SCALING FITS (log score ~ a + b*logN + c*logE + d*logN*logE):")
    lines.append(FIT.to_string(index=False))
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  b_logN: How score scales with system size")
    lines.append("  c_logE: How score scales with coupling strength")
    lines.append("  d_logNlogE: Interaction (scale-dependent coupling)")
    
    txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[PASS36] wrote {p_pairs}")
    print(f"[PASS36] wrote {p_inv}")
    print(f"[PASS36] wrote {p_fit}")
    print(f"[PASS36] wrote {txt}")


if __name__ == "__main__":
    main()
