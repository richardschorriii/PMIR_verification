#!/usr/bin/env python3
"""
PASS30 - PMIR Verification Test
Extracted from ChatGPT transcript (line 70576)
Length: 7722 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope of y ~ b0 + b1*x via OLS."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom <= 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / denom)


def _safe_log(v: np.ndarray, floor: float) -> np.ndarray:
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.nan)
    # enforce positivity for log
    v = np.where(v > floor, v, floor)
    return np.log(v)


def bootstrap_slope_stratN(
    df: pd.DataFrame,
    topology: str,
    ycol: str,
    reps: int,
    seed: int,
    floor: float,
) -> dict:
    base = df[df["topology"] == topology].copy()
    base = base[np.isfinite(base["N"])].copy()
    base["N"] = base["N"].astype(int)

    Ns = sorted(base["N"].unique().tolist())
    # Need at least 3 distinct N to define slope robustly
    if len(Ns) < 3:
        return {
            "topology": topology,
            "metric_col": ycol,
            "graphs_total": int(base["graph_seed"].nunique()) if "graph_seed" in base.columns else np.nan,
            "Ns": ",".join(map(str, Ns)),
            "alpha_hat": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_boot_two_sided": np.nan,
            "status": "insufficient_N",
        }

    # group indices by N
    idx_by_N = {}
    for N in Ns:
        idx = base.index[base["N"] == N].to_numpy()
        idx_by_N[int(N)] = idx

    # compute point estimate using mean across graphs at each N
    y_means = []
    for N in Ns:
        vals = base.loc[idx_by_N[int(N)], ycol].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            y_means.append(np.nan)
        else:
            y_means.append(float(np.mean(vals)))
    y_means = np.array(y_means, float)

    if not np.all(np.isfinite(y_means)):
        return {
            "topology": topology,
            "metric_col": ycol,
            "graphs_total": int(base["graph_seed"].nunique()) if "graph_seed" in base.columns else np.nan,
            "Ns": ",".join(map(str, Ns)),
            "alpha_hat": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_boot_two_sided": np.nan,
            "status": "nan_means",
        }

    x = _safe_log(np.array(Ns, float), floor=floor)
    y = _safe_log(y_means, floor=floor)
    alpha_hat = _ols_slope(x, y)

    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(reps), float)
    boot[:] = np.nan

    for r in range(int(reps)):
        bm = []
        for N in Ns:
            idx = idx_by_N[int(N)]
            # sample graphs within N with replacement
            pick = rng.choice(idx, size=len(idx), replace=True)
            vals = base.loc[pick, ycol].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                bm.append(np.nan)
            else:
                bm.append(float(np.mean(vals)))
        bm = np.array(bm, float)
        if not np.all(np.isfinite(bm)):
            boot[r] = np.nan
            continue
        yb = _safe_log(bm, floor=floor)
        boot[r] = _ols_slope(x, yb)

    boot = boot[np.isfinite(boot)]
    if len(boot) < max(100, reps // 10):
        return {
            "topology": topology,
            "metric_col": ycol,
            "graphs_total": int(base["graph_seed"].nunique()) if "graph_seed" in base.columns else np.nan,
            "Ns": ",".join(map(str, Ns)),
            "alpha_hat": float(alpha_hat) if np.isfinite(alpha_hat) else np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_boot_two_sided": np.nan,
            "status": "boot_too_few",
        }

    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975]).tolist()

    # two-sided bootstrap sign p-value (simple, stable)
    if np.isfinite(alpha_hat) and alpha_hat != 0:
        frac_opposite = float(np.mean(np.sign(boot) != np.sign(alpha_hat)))
        p2 = min(1.0, 2.0 * frac_opposite)
    else:
        # if alpha_hat==0, use centered mass near 0 as "p"
        p2 = float(np.mean(np.abs(boot) >= np.abs(alpha_hat)))

    return {
        "topology": topology,
        "metric_col": ycol,
        "graphs_total": int(base["graph_seed"].nunique()) if "graph_seed" in base.columns else np.nan,
        "Ns": ",".join(map(str, Ns)),
        "alpha_hat": float(alpha_hat),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_boot_two_sided": float(p2),
        "boot_n": int(len(boot)),
        "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topologies", default="rr,grid2d_periodic")
    ap.add_argument("--metrics", default="std_F,stability_ratio,mean_abs_F,sign_flip_rate")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log_floor", type=float, default=1e-12)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # numeric hygiene
    for c in ["N", "graph_seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    rows = []
    for topo in topologies:
        for m in metrics:
            if m not in df.columns:
                rows.append(
                    {
                        "topology": topo,
                        "metric_col": m,
                        "status": "missing_metric",
                        "alpha_hat": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "p_boot_two_sided": np.nan,
                    }
                )
                continue
            rows.append(
                bootstrap_slope_stratN(
                    df=df,
                    topology=topo,
                    ycol=m,
                    reps=args.reps,
                    seed=args.seed,
                    floor=args.log_floor,
                )
            )

    out = pd.DataFrame(rows)
    out_csv = outdir / "pass30_bootstrap_slopes.csv"
    out.to_csv(out_csv, index=False)

    # write a compact txt summary
    lines = []
    lines.append("PASS 30 — Bootstrap scaling slope CIs (log metric vs log N), stratified by N")
    lines.append(f"in: {args.in_csv}")
    lines.append(f"reps: {args.reps}  seed: {args.seed}  log_floor: {args.log_floor:g}")
    lines.append("")
    if len(out):
        show = out[["topology", "metric_col", "alpha_hat", "ci_lo", "ci_hi", "p_boot_two_sided", "status"]].copy()
        lines.append(show.to_string(index=False))
    txt = "\n".join(lines)
    out_txt = outdir / "pass30_bootstrap_slopes.txt"
    out_txt.write_text(txt, encoding="utf-8")

    print(f"[PASS30] wrote {out_csv}")
    print(f"[PASS30] wrote {out_txt}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
Run it (PowerShell)
$IN30  = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass20_seed_stability\pass20_seed_stability_by_graph.csv"
$O30   = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass30_bootstrap_scaling_ci"
mkdir $O30 -Force | Out-Null

python .\pass30_bootstrap_scaling_ci.py `
  --in_csv  $IN30 `
  --outdir  $O30 `
  --reps    20000 `
  --seed    1337
What to look for
For each (topology, metric): ci_lo and ci_hi