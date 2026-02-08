#!/usr/bin/env python3
"""
PASS32 - PMIR Verification Test
Extracted from ChatGPT transcript (line 78825)
Length: 7196 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def perm_delta_mean_stratN(df, topo_a, topo_b, Ncol, ycol, reps=10000, seed=1337):
    """
    Two-sided permutation test for delta_mean = mean(a)-mean(b),
    stratified by N (permute topology labels within each N).
    df is expected to already be graph-aggregated (one row per graph).
    """
    rng = np.random.default_rng(seed)

    base = df[df["topology"].isin([topo_a, topo_b])].copy()
    base = base.dropna(subset=[Ncol, ycol, "topology"])
    if len(base) < 4:
        return dict(p=np.nan, delta=np.nan, reps=reps, seed=seed, status="too_few")

    A = base[base.topology == topo_a][ycol].to_numpy()
    B = base[base.topology == topo_b][ycol].to_numpy()
    if len(A) == 0 or len(B) == 0:
        return dict(p=np.nan, delta=np.nan, reps=reps, seed=seed, status="missing_group")

    delta_obs = float(np.mean(A) - np.mean(B))

    groups = {}
    for N, idx in base.groupby(Ncol).indices.items():
        groups[int(N)] = np.asarray(idx, dtype=int)

    topo = base["topology"].to_numpy()
    y = base[ycol].to_numpy()

    deltas = np.zeros(reps, dtype=float)
    for r in range(reps):
        topo_perm = topo.copy()
        for _, idx in groups.items():
            topo_perm[idx] = rng.permutation(topo_perm[idx])
        deltas[r] = np.mean(y[topo_perm == topo_a]) - np.mean(y[topo_perm == topo_b])

    p = float((np.sum(np.abs(deltas) >= abs(delta_obs)) + 1) / (reps + 1))
    return dict(p=p, delta=delta_obs, reps=reps, seed=seed, status="ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score_col", default="auc_delta")   # use AUC as main response
    ap.add_argument("--topo_a", default="rr")
    ap.add_argument("--topo_b", default="grid2d_periodic")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--agg", default="mean", choices=["mean", "median"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.in_csv)

    # numeric hygiene
    for c in ["N", "graph_seed", "pmir_seed", "probe_eps", args.score_col, "censored", "lambda2_base"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # require P2 columns
    need = {"pass_id", "topology", "N", "graph_seed", "pmir_seed", "probe_eps",
            "probe_dir_mode", "probe_mode", args.score_col}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing required columns: {miss}")

    d = df[df["pass_id"] == "P2_PROBE_INVARIANCE"].copy()
    if len(d) == 0:
        raise SystemExit("[ERR] No rows with pass_id==P2_PROBE_INVARIANCE. Run pmir_master_time_lock.py --do_probe_invariance first.")

    d = d.dropna(subset=["topology", "N", "graph_seed", "pmir_seed", "probe_eps",
                         "probe_dir_mode", "probe_mode", args.score_col])

    # Aggregate over pmir_seed => per (topology,N,graph_seed,eps,dir_mode,probe_mode)
    gcols = ["topology", "N", "graph_seed", "probe_eps", "probe_dir_mode", "probe_mode"]

    if args.agg == "mean":
        aggF = {
            args.score_col: "mean",
            "pmir_seed": "nunique",
            "censored": "mean" if "censored" in d.columns else "size",
        }
    else:
        aggF = {
            args.score_col: "median",
            "pmir_seed": "nunique",
            "censored": "mean" if "censored" in d.columns else "size",
        }

    G = (d.groupby(gcols, as_index=False)
           .agg(aggF)
           .rename(columns={
                args.score_col: "score_mean",
                "pmir_seed": "pmir_seeds",
                "censored": "censored_frac"
           }))

    # within-graph std over pmir_seed
    S = d.groupby(gcols)[args.score_col].std(ddof=1).reset_index().rename(columns={args.score_col: "score_std"})
    G = G.merge(S, on=gcols, how="left")

    p_graph = outdir / "pass33_by_graph_eps_dir_probe.csv"
    G.to_csv(p_graph, index=False)

    # Summary by (dir_mode,probe_mode,topology,N,eps)
    H = (G.groupby(["probe_dir_mode", "probe_mode", "topology", "N", "probe_eps"], as_index=False)
           .agg(graphs=("graph_seed", "nunique"),
                mean_score=("score_mean", "mean"),
                mean_std=("score_std", "mean"),
                mean_censored=("censored_frac", "mean")))
    p_sum = outdir / "pass33_summary_by_dir_probe_topoN_eps.csv"
    H.to_csv(p_sum, index=False)

    # Contrasts rr vs grid for each (dir_mode,probe_mode,eps), stratified by N
    rows = []
    for (dm, pm, eps), sub in G.groupby(["probe_dir_mode", "probe_mode", "probe_eps"]):
        perm = perm_delta_mean_stratN(sub, args.topo_a, args.topo_b, "N", "score_mean",
                                      reps=args.reps, seed=args.seed)
        rows.append(dict(
            probe_dir_mode=str(dm),
            probe_mode=str(pm),
            probe_eps=float(eps),
            topo_a=args.topo_a,
            topo_b=args.topo_b,
            delta_mean_a_minus_b=perm["delta"],
            p_perm_two_sided_stratified_by_N=perm["p"],
            reps=args.reps,
            seed=args.seed,
            status=perm["status"],
            n_rows=int(len(sub)),
        ))

    C = pd.DataFrame(rows).sort_values(["probe_dir_mode", "probe_mode", "probe_eps"])
    p_con = outdir / "pass33_contrast_by_dir_probe_eps.csv"
    C.to_csv(p_con, index=False)

    # quick text
    txt = outdir / "pass33_probe_invariance_auc.txt"
    lines = []
    lines.append("PASS33 — probe invariance (AUC, graph-aggregated over pmir_seed)")
    lines.append(f"in: {args.in_csv}")
    lines.append(f"score_col: {args.score_col}  agg={args.agg}")
    lines.append(f"wrote: {p_graph}")
    lines.append(f"wrote: {p_sum}")
    lines.append(f"wrote: {p_con}")
    lines.append("")
    lines.append("contrast per (dir_mode, probe_mode, eps): delta(rr-grid), perm p")
    for (dm, pm), sub in C.groupby(["probe_dir_mode", "probe_mode"]):
        lines.append(f"--- dir={dm}  probe_mode={pm} ---")
        for _, r in sub.iterrows():
            lines.append(f"  eps={r.probe_eps:.5f}  delta={r.delta_mean_a_minus_b:+.6e}  p={r.p_perm_two_sided_stratified_by_N:.6g}  n={int(r.n_rows)}")
    txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[PASS33] wrote {p_graph}")
    print(f"[PASS33] wrote {p_sum}")
    print(f"[PASS33] wrote {p_con}")
    print(f"[PASS33] wrote {txt}")


if __name__ == "__main__":
    main()
2) PowerShell-safe commands (run PASS33)
If you already ran P2_PROBE_INVARIANCE and it’s inside your enriched master:
$M_ENRICH = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\baseline\tauhalf_master_v1.csv"
$O33 = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\baseline\pass33_probe_invariance_auc"

python .\pass33_probe_invariance_auc.py `
  --in_csv  $M_ENRICH `
  --outdir  $O33 `
  --score_col auc_delta `
  --topo_a rr --topo_b grid2d_periodic `
  --reps 10000 --seed 1337 `
  --agg mean