#!/usr/bin/env python3
"""
PASS28 - PMIR Verification Test
Extracted from ChatGPT transcript (line 69754)
Length: 8970 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_METRICS = ["mean_abs_F", "std_F", "sign_flip_rate", "stability_ratio"]
GRID_NAMES = {"grid2d_periodic", "grid2d_periodic_mix", "grid"}  # accept variants


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    # d = (mean(x)-mean(y))/pooled_sd
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def auroc_rank(x: np.ndarray, y: np.ndarray) -> float:
    """
    AUROC from rank statistic (equiv to Mann–Whitney U).
    Returns P(X > Y) with ties=0.5.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    z = np.concatenate([x, y])
    ranks = pd.Series(z).rank(method="average").to_numpy()
    rx = ranks[:nx].sum()
    Ux = rx - nx * (nx + 1) / 2.0
    return Ux / (nx * ny)


def stratified_permutation_pvalue(df: pd.DataFrame,
                                  metric: str,
                                  reps: int,
                                  rng: np.random.Generator) -> dict:
    """
    Stratify by N; shuffle topology labels within each N.
    Test statistic: delta_mean = mean(rr) - mean(grid), pooled across strata
                   via simple concatenation (since strata sizes equal here).
    """
    d = df[["N", "is_rr", metric]].dropna().copy()
    if d.empty:
        return {"p_perm": np.nan, "delta_perm_mean": np.nan}

    # observed
    rr = d[d.is_rr == 1][metric].to_numpy()
    gg = d[d.is_rr == 0][metric].to_numpy()
    delta_obs = float(np.mean(rr) - np.mean(gg))

    # prep indices per stratum
    strata = {}
    for Nval, g in d.groupby("N"):
        idx = g.index.to_numpy()
        strata[int(Nval)] = idx

    deltas = np.empty(reps, dtype=float)

    is_rr = d.is_rr.to_numpy().astype(int)
    vals = d[metric].to_numpy(dtype=float)

    # map from original df index to position in arrays
    # (d is a filtered copy; we use positional arrays)
    pos = {ix: i for i, ix in enumerate(d.index.to_numpy())}

    strata_pos = {Nval: np.array([pos[ix] for ix in idx], dtype=int)
                  for Nval, idx in strata.items()}

    for r in range(reps):
        labels = is_rr.copy()
        for Nval, pidx in strata_pos.items():
            # shuffle labels within this stratum
            labels[pidx] = rng.permutation(labels[pidx])
        rr_m = vals[labels == 1]
        gg_m = vals[labels == 0]
        deltas[r] = np.mean(rr_m) - np.mean(gg_m)

    # two-sided p-value
    p = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1.0) / (reps + 1.0)
    return {"p_perm": float(p), "delta_obs": delta_obs,
            "delta_perm_mean": float(np.mean(deltas)), "delta_perm_std": float(np.std(deltas, ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_graph_csv", required=True, help="pass20_seed_stability_by_graph.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rr_name", default="rr")
    ap.add_argument("--grid_name", default="grid2d_periodic")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_graph_csv)

    # Basic hygiene
    need_cols = ["topology", "N", "graph_seed"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"[ERR] missing required column: {c}")

    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df.dropna(subset=["N"]).copy()
    df["N"] = df["N"].astype(int)

    # Identify rr vs grid
    topo = df["topology"].astype(str)
    is_rr = (topo.str.lower() == args.rr_name.lower())
    is_grid = topo.str.lower().isin({args.grid_name.lower(), *GRID_NAMES})
    df = df[is_rr | is_grid].copy()
    df["is_rr"] = (df["topology"].astype(str).str.lower() == args.rr_name.lower()).astype(int)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] missing metric columns: {missing}")

    rng = np.random.default_rng(args.seed)

    # ---- By N summaries ----
    rows_byN = []
    for Nval in sorted(df["N"].unique().tolist()):
        g = df[df["N"] == Nval].copy()
        # require both classes present
        if g["is_rr"].nunique() < 2:
            continue
        for m in metrics:
            x = g[g.is_rr == 1][m].to_numpy(dtype=float)
            y = g[g.is_rr == 0][m].to_numpy(dtype=float)

            delta = float(np.mean(x) - np.mean(y))
            dcohen = float(cohen_d(x, y))
            au = float(auroc_rank(x, y))

            perm = stratified_permutation_pvalue(g, m, reps=args.reps, rng=rng)

            rows_byN.append({
                "N": int(Nval),
                "metric": m,
                "n_rr": int(np.sum(g.is_rr == 1)),
                "n_grid": int(np.sum(g.is_rr == 0)),
                "mean_rr": float(np.mean(x)),
                "mean_grid": float(np.mean(y)),
                "delta_mean_rr_minus_grid": delta,
                "cohen_d": dcohen,
                "auroc_rr_gt_grid": au,
                "p_perm_two_sided": perm["p_perm"],
                "perm_delta_mean": perm["delta_perm_mean"],
                "perm_delta_std": perm["delta_perm_std"],
            })

    byN = pd.DataFrame(rows_byN)
    byN_path = outdir / "pass28_topology_contrast_byN.csv"
    byN.to_csv(byN_path, index=False)

    # ---- Overall (stratified by N) ----
    rows_overall = []
    for m in metrics:
        perm = stratified_permutation_pvalue(df, m, reps=args.reps, rng=rng)

        rr = df[df.is_rr == 1][m].to_numpy(dtype=float)
        gg = df[df.is_rr == 0][m].to_numpy(dtype=float)

        rows_overall.append({
            "metric": m,
            "n_rr": int(np.sum(df.is_rr == 1)),
            "n_grid": int(np.sum(df.is_rr == 0)),
            "mean_rr": float(np.mean(rr)),
            "mean_grid": float(np.mean(gg)),
            "delta_mean_rr_minus_grid": float(np.mean(rr) - np.mean(gg)),
            "cohen_d": float(cohen_d(rr, gg)),
            "auroc_rr_gt_grid": float(auroc_rank(rr, gg)),
            "p_perm_two_sided_stratified_by_N": perm["p_perm"],
            "perm_delta_mean": perm["delta_perm_mean"],
            "perm_delta_std": perm["delta_perm_std"],
            "reps": int(args.reps),
            "seed": int(args.seed),
        })

    overall = pd.DataFrame(rows_overall)
    overall_path = outdir / "pass28_topology_contrast_overall.csv"
    overall.to_csv(overall_path, index=False)

    # ---- Text report ----
    txt = []
    txt.append("PASS 28 — Topology contrast test (graph-level; stratified by N)")
    txt.append(f"in: {args.in_graph_csv}")
    txt.append(f"metrics: {', '.join(metrics)}")
    txt.append(f"reps: {args.reps}  seed: {args.seed}")
    txt.append("")
    txt.append(f"rows(in, filtered): {len(df)}")
    txt.append(f"N values: {sorted(df.N.unique().tolist())}")
    txt.append(f"graphs per (topology,N):")
    txt.append(df.groupby(["topology", "N"])["graph_seed"].nunique().reset_index(name="graphs").to_string(index=False))
    txt.append("")
    if not byN.empty:
        txt.append("By-N results (delta=mean(rr)-mean(grid); p_perm is stratified within this N only):")
        txt.append(byN.sort_values(["metric", "N"]).to_string(index=False))
        txt.append("")
    txt.append("Overall stratified-by-N permutation results:")
    txt.append(overall.to_string(index=False))
    txt.append("")

    txt_path = outdir / "pass28_topology_contrast.txt"
    txt_path.write_text("\n".join(txt), encoding="utf-8")

    print(f"[PASS28] wrote {byN_path}")
    print(f"[PASS28] wrote {overall_path}")
    print(f"[PASS28] wrote {txt_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
Run command (PowerShell)
$IN28  = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass20_seed_stability\pass20_seed_stability_by_graph.csv"
$O28   = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass28_topology_contrast"
mkdir $O28 -Force | Out-Null

python .\pass28_topology_contrast.py `
  --in_graph_csv $IN28 `
  --outdir $O28 `
  --metrics mean_abs_F,std_F,sign_flip_rate,stability_ratio `
  --reps 10000 --seed 1337
If you want PASS28 to operate on sample-level (900 rows) instead of graph-level, say “PASS28 sample-level” and I’ll drop a variant that stratifies on (N, graph_seed) and uses the pass19_pmir_field_index.csv panel.