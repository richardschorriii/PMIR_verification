#!/usr/bin/env python3
"""
PASS31 - PMIR Verification Test
Extracted from ChatGPT transcript (line 71197)
Length: 10285 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _ols_fit(x, y):
    """Return (slope, intercept, r2) for y = slope*x + intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    if np.allclose(x, x[0]):
        # constant x => cannot fit slope
        return np.nan, np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef[0], coef[1]
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def _quad_fit(x, y):
    """Return (curv_c, slope_a, intercept_b, r2) for y = c*x^2 + a*x + b."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    if np.allclose(x, x[0]):
        return np.nan, np.nan, np.nan, np.nan
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, a, b = coef[0], coef[1], coef[2]
    yhat = c * x**2 + a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    return float(c), float(a), float(b), float(r2)


def _perm_contrast_stratN(df, topo_a, topo_b, Ncol, valcol, reps=10000, seed=1337):
    """
    Stratified permutation test for mean difference of valcol (topo_a - topo_b),
    where the unit is a row in df and we shuffle labels within each N stratum.
    """
    rng = np.random.default_rng(seed)
    base = df[[Ncol, "topology", valcol]].dropna().copy()
    base = base[base["topology"].isin([topo_a, topo_b])]
    if base.empty:
        return {"status": "empty", "p_two": np.nan, "obs_delta": np.nan, "perm_mean": np.nan, "perm_std": np.nan}

    # observed delta (mean topo_a - mean topo_b), stratified by N (weighted by stratum size)
    obs_parts = []
    for N, g in base.groupby(Ncol):
        a = g[g["topology"] == topo_a][valcol].to_numpy()
        b = g[g["topology"] == topo_b][valcol].to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        obs_parts.append(np.mean(a) - np.mean(b))
    if len(obs_parts) == 0:
        return {"status": "missing_strata", "p_two": np.nan, "obs_delta": np.nan, "perm_mean": np.nan, "perm_std": np.nan}
    obs_delta = float(np.mean(obs_parts))

    # indices per stratum (note: groupby.indices gives ndarray already)
    groups = {int(N): idx for N, idx in base.groupby(Ncol).indices.items()}

    vals = base[valcol].to_numpy()
    tops = base["topology"].to_numpy()

    perm_deltas = np.empty(reps, float)
    for r in range(reps):
        deltas = []
        for N, idx in groups.items():
            t = tops[idx].copy()
            rng.shuffle(t)
            a = vals[idx][t == topo_a]
            b = vals[idx][t == topo_b]
            if len(a) == 0 or len(b) == 0:
                continue
            deltas.append(np.mean(a) - np.mean(b))
        perm_deltas[r] = np.mean(deltas) if len(deltas) else np.nan

    perm_deltas = perm_deltas[np.isfinite(perm_deltas)]
    if len(perm_deltas) == 0:
        return {"status": "perm_failed", "p_two": np.nan, "obs_delta": obs_delta, "perm_mean": np.nan, "perm_std": np.nan}

    p_two = float((np.sum(np.abs(perm_deltas) >= abs(obs_delta)) + 1) / (len(perm_deltas) + 1))
    return {
        "status": "ok",
        "p_two": p_two,
        "obs_delta": obs_delta,
        "perm_mean": float(np.mean(perm_deltas)),
        "perm_std": float(np.std(perm_deltas, ddof=1)) if len(perm_deltas) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Panel/feature table containing probe_eps and score column.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score_col", default="PMIR_F", help="Score to fit vs eps (default: PMIR_F).")
    ap.add_argument("--topo_a", default="rr")
    ap.add_argument("--topo_b", default="grid2d_periodic")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    # required columns
    need = ["topology", "N", "graph_seed", "probe_eps", args.score_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] missing required columns: {missing}")

    # numeric hygiene
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["graph_seed"] = pd.to_numeric(df["graph_seed"], errors="coerce")
    df["probe_eps"] = pd.to_numeric(df["probe_eps"], errors="coerce")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")

    # We allow pmir_seed to exist (recommended) but not required for graph-level fits.
    if "pmir_seed" in df.columns:
        df["pmir_seed"] = pd.to_numeric(df["pmir_seed"], errors="coerce")

    # graph-level: fit score vs eps per (topology,N,graph_seed)
    rows = []
    grp_cols = ["topology", "N", "graph_seed"]
    for key, g in df.groupby(grp_cols):
        topo, N, gs = key
        x = g["probe_eps"].to_numpy()
        y = g[args.score_col].to_numpy()
        eps_unique = np.unique(x[np.isfinite(x)])
        eps_count = len(eps_unique)

        slope, intercept, r2_lin = _ols_fit(x, y)
        curv, slope_q, intercept_q, r2_quad = _quad_fit(x, y) if eps_count >= 3 else (np.nan, np.nan, np.nan, np.nan)

        rows.append({
            "topology": topo,
            "N": int(N) if np.isfinite(N) else np.nan,
            "graph_seed": int(gs) if np.isfinite(gs) else np.nan,
            "eps_count": int(eps_count),
            "eps_min": float(np.nanmin(eps_unique)) if eps_count else np.nan,
            "eps_max": float(np.nanmax(eps_unique)) if eps_count else np.nan,
            "slope_lin": slope,
            "intercept_lin": intercept,
            "r2_lin": r2_lin,
            "curv_quad": curv,          # c in c*eps^2 + a*eps + b
            "slope_quad": slope_q,      # a
            "intercept_quad": intercept_q,
            "r2_quad": r2_quad,
        })

    by_graph = pd.DataFrame(rows)
    p_by_graph = outdir / "pass31_eps_response_by_graph.csv"
    by_graph.to_csv(p_by_graph, index=False)

    # aggregate by (topology,N)
    agg = (by_graph
           .groupby(["topology", "N"], as_index=False)
           .agg(
               graphs=("graph_seed", "nunique"),
               eps_count_med=("eps_count", "median"),
               slope_lin_mean=("slope_lin", "mean"),
               slope_lin_std=("slope_lin", "std"),
               r2_lin_mean=("r2_lin", "mean"),
               curv_quad_mean=("curv_quad", "mean"),
               curv_quad_std=("curv_quad", "std"),
               r2_quad_mean=("r2_quad", "mean"),
           ))
    p_by_N = outdir / "pass31_eps_response_by_N.csv"
    agg.to_csv(p_by_N, index=False)

    # topology contrast (slope_lin and curv_quad), stratified by N
    contrast_rows = []
    for col in ["slope_lin", "curv_quad"]:
        tmp = by_graph[by_graph["topology"].isin([args.topo_a, args.topo_b])].copy()
        # only compare when the column is finite for both topologies in at least one stratum
        perm = _perm_contrast_stratN(tmp, args.topo_a, args.topo_b, "N", col, reps=args.reps, seed=args.seed)
        contrast_rows.append({
            "metric": col,
            "topo_a": args.topo_a,
            "topo_b": args.topo_b,
            "delta_mean_a_minus_b": perm["obs_delta"],
            "p_perm_two_sided_stratified_by_N": perm["p_two"],
            "perm_delta_mean": perm["perm_mean"],
            "perm_delta_std": perm["perm_std"],
            "reps": args.reps,
            "seed": args.seed,
            "status": perm["status"],
        })
    contrast = pd.DataFrame(contrast_rows)
    p_contrast = outdir / "pass31_eps_response_topo_contrast.csv"
    contrast.to_csv(p_contrast, index=False)

    # text summary
    lines = []
    lines.append("PASS 31 — ε-response (linearity / curvature) audit")
    lines.append(f"in: {args.in_csv}")
    lines.append(f"score_col: {args.score_col}")
    lines.append(f"rows(in): {len(df)}   rows(graph-fits): {len(by_graph)}")
    lines.append("")
    lines.append("Aggregate by topology,N:")
    lines.append(agg.to_string(index=False))
    lines.append("")
    lines.append("Topology contrast (stratified by N):")
    lines.append(contrast.to_string(index=False))
    lines.append("")
    # detect if only one eps overall
    eps_unique_all = sorted(df["probe_eps"].dropna().unique().tolist())
    lines.append(f"probe_eps unique (global): {eps_unique_all}")
    if len(eps_unique_all) < 2:
        lines.append("[NOTE] Only 1 eps present => cannot test linearity; rerun PASS31 after multi-eps sweep.")
    elif len(eps_unique_all) < 3:
        lines.append("[NOTE] Only 2 eps present => linear slope only; need ≥3 eps for curvature (quadratic) test.")

    p_txt = outdir / "pass31_eps_response.txt"
    p_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[PASS31] wrote {p_by_graph}")
    print(f"[PASS31] wrote {p_by_N}")
    print(f"[PASS31] wrote {p_contrast}")
    print(f"[PASS31] wrote {p_txt}")


if __name__ == "__main__":
    main()
Run command (your current layout)
Use PASS19 output as input (it already has probe_eps, topology, N, graph_seed, pmir_seed, PMIR_F, etc.):

$P19 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass19_pmir_field\pass19_pmir_field_index.csv"
$O31 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass31_eps_response"
mkdir $O31 -Force | Out-Null

python .\pass31_eps_response.py `
  --in_csv  $P19 `
  --outdir  $O31 `
  --score_col PMIR_F `
  --topo_a rr --topo_b grid2d_periodic `
  --reps 10000 --seed 1337
What you should expect right now (given only eps=0.05)
Script will run and write outputs.