#!/usr/bin/env python3
"""
PASS26 - PMIR Verification Test
Extracted from ChatGPT transcript (line 70151)
Length: 8497 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _resolve_metric_col(df: pd.DataFrame, want: str) -> str:
    """
    Try hard to find the intended metric column.
    Accepts raw, *_mean, etc.
    """
    cands = [
        want,
        f"{want}_mean",
        f"{want}_med",
        f"{want}_median",
        f"{want}_std",
        f"{want}_mean_mean",
        f"{want}_mean_med",
        f"{want}_mean_std",
        # common pass20 naming seen in your printouts:
        f"{want}_mean",
        f"{want}_med",
    ]
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a column for metric='{want}'. Tried: {cands}. Available: {df.columns.tolist()}")


def _ols_loglog(xN: np.ndarray, y: np.ndarray):
    """
    Fit: log(y) = a*log(N) + b  (with y>0)
    Returns dict with slope/intercept/r2/se_slope/n/status.
    """
    out = dict(status="ok", n=int(len(y)))
    m = np.isfinite(xN) & np.isfinite(y) & (xN > 0) & (y > 0)
    x = np.log(xN[m].astype(float))
    yy = np.log(y[m].astype(float))

    if len(yy) < 3 or np.allclose(x.var(), 0.0):
        out.update(status="fit_failed", slope=np.nan, intercept=np.nan, r2=np.nan, se_slope=np.nan, n=int(len(yy)))
        return out

    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    yhat = X @ beta
    resid = yy - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst

    # standard error for slope
    n = len(yy)
    dof = max(n - 2, 1)
    sigma2 = sse / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_a = float(np.sqrt(sigma2 * xtx_inv[0, 0]))

    out.update(slope=a, intercept=b, r2=r2, se_slope=se_a, n=int(n))
    return out


def _slope_for_topology(df: pd.DataFrame, topo: str, Ncol: str, ycol: str):
    g = df[df["topology"].astype(str) == topo].copy()
    if len(g) == 0:
        return dict(status="no_rows", slope=np.nan, intercept=np.nan, r2=np.nan, se_slope=np.nan, n=0)
    return _ols_loglog(g[Ncol].to_numpy(), g[ycol].to_numpy())


def _perm_slope_contrast_stratN(df: pd.DataFrame, topo_a: str, topo_b: str, Ncol: str, ycol: str, reps: int, seed: int):
    """
    Permute topology labels *within each N* to preserve N composition.
    Contrast = slope(topo_a) - slope(topo_b).
    """
    rng = np.random.default_rng(seed)

    base = df[[Ncol, "topology", ycol]].copy()
    base["topology"] = base["topology"].astype(str)

    # keep only the two groups
    base = base[base["topology"].isin([topo_a, topo_b])].copy()

    # observed
    fa = _slope_for_topology(base, topo_a, Ncol, ycol)
    fb = _slope_for_topology(base, topo_b, Ncol, ycol)
    if fa["status"] != "ok" or fb["status"] != "ok":
        return dict(status="fit_failed", obs_delta=np.nan, p_perm=np.nan, reps=reps, seed=seed)

    obs = fa["slope"] - fb["slope"]

    deltas = np.empty(reps, dtype=float)
    deltas.fill(np.nan)

    # pre-split indices by N for stratified shuffle
    groups = {int(N): idx.to_numpy() for N, idx in base.groupby(Ncol).indices.items()}

    topo_arr = base["topology"].to_numpy()

    for r in range(reps):
        perm_topo = topo_arr.copy()
        for N, idxs in groups.items():
            perm_topo[idxs] = rng.permutation(perm_topo[idxs])
        tmp = base.copy()
        tmp["topology"] = perm_topo

        pa = _slope_for_topology(tmp, topo_a, Ncol, ycol)
        pb = _slope_for_topology(tmp, topo_b, Ncol, ycol)
        if pa["status"] == "ok" and pb["status"] == "ok":
            deltas[r] = pa["slope"] - pb["slope"]

    good = np.isfinite(deltas)
    if good.sum() < max(100, reps // 10):
        return dict(status="perm_failed", obs_delta=obs, p_perm=np.nan, reps=int(good.sum()), seed=seed)

    # two-sided p-value
    p = (np.sum(np.abs(deltas[good]) >= abs(obs)) + 1.0) / (good.sum() + 1.0)
    return dict(status="ok", obs_delta=obs, p_perm=float(p), reps=int(good.sum()), seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metrics", default="std_F,stability_ratio",
                    help="Comma-separated metric base names (will auto-resolve actual col names).")
    ap.add_argument("--topo_a", default="rr")
    ap.add_argument("--topo_b", default="grid2d_periodic")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # basic hygiene
    if "topology" not in df.columns:
        raise KeyError("Expected column 'topology' in input.")
    if "N" not in df.columns:
        # sometimes 'N' is string; try to resolve
        raise KeyError("Expected column 'N' in input.")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df.dropna(subset=["N", "topology"]).copy()
    df["N"] = df["N"].astype(int)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    fit_rows = []
    contrast_rows = []
    txt_lines = []
    txt_lines.append("PASS 29 — Topology–N scaling collapse (log–log)")
    txt_lines.append(f"in: {in_path}")
    txt_lines.append(f"rows(in): {len(df)}")
    txt_lines.append(f"topologies present: {sorted(df['topology'].astype(str).unique().tolist())}")
    txt_lines.append(f"N present: {sorted(df['N'].unique().tolist())}")
    txt_lines.append("")

    for m in metrics:
        ycol = _resolve_metric_col(df, m)

        fa = _slope_for_topology(df, args.topo_a, "N", ycol)
        fb = _slope_for_topology(df, args.topo_b, "N", ycol)

        fit_rows.append(dict(metric=m, metric_col=ycol, topology=args.topo_a, **fa))
        fit_rows.append(dict(metric=m, metric_col=ycol, topology=args.topo_b, **fb))

        perm = _perm_slope_contrast_stratN(df, args.topo_a, args.topo_b, "N", ycol, args.reps, args.seed)
        contrast_rows.append(dict(
            metric=m,
            metric_col=ycol,
            topo_a=args.topo_a,
            topo_b=args.topo_b,
            slope_a=fa.get("slope", np.nan),
            slope_b=fb.get("slope", np.nan),
            delta_slope_a_minus_b=(fa.get("slope", np.nan) - fb.get("slope", np.nan)),
            p_perm_two_sided_stratified_by_N=perm.get("p_perm", np.nan),
            reps=perm.get("reps", args.reps),
            seed=args.seed,
            status=perm.get("status", "unknown"),
        ))

        txt_lines.append(f"Metric: {m}  (using col='{ycol}')")
        txt_lines.append(f"  {args.topo_a}: slope={fa.get('slope',np.nan): .6f}  se={fa.get('se_slope',np.nan): .6f}  r2={fa.get('r2',np.nan): .4f}  n={fa.get('n',0)}  status={fa.get('status')}")
        txt_lines.append(f"  {args.topo_b}: slope={fb.get('slope',np.nan): .6f}  se={fb.get('se_slope',np.nan): .6f}  r2={fb.get('r2',np.nan): .4f}  n={fb.get('n',0)}  status={fb.get('status')}")
        txt_lines.append(f"  slope contrast ({args.topo_a}-{args.topo_b}) = {contrast_rows[-1]['delta_slope_a_minus_b']:.6f}   perm p={contrast_rows[-1]['p_perm_two_sided_stratified_by_N']}")
        txt_lines.append("")

    fit_df = pd.DataFrame(fit_rows)
    con_df = pd.DataFrame(contrast_rows)

    fit_csv = outdir / "pass29_scaling_by_topology.csv"
    con_csv = outdir / "pass29_slope_contrast.csv"
    txt_path = outdir / "pass29_scaling_collapse.txt"

    fit_df.to_csv(fit_csv, index=False)
    con_df.to_csv(con_csv, index=False)
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("[PASS29] wrote", fit_csv)
    print("[PASS29] wrote", con_csv)
    print("[PASS29] wrote", txt_path)
    print(con_df.to_string(index=False))


if __name__ == "__main__":
    main()
Run it (PowerShell)
$IN29  = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass20_seed_stability\pass20_seed_stability_by_graph.csv"
$O29   = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass29_scaling_collapse"
mkdir $O29 -Force | Out-Null

python .\pass29_scaling_collapse.py `
  --in_csv $IN29 `
  --outdir $O29 `
  --metrics "std_F,stability_ratio,mean_abs_F,sign_flip_rate" `
  --topo_a rr `
  --topo_b grid2d_periodic `
  --reps 10000 `
  --seed 1337
What PASS29 will tell you
For each metric, it fits log(metric) ~ slope*log(N) + intercept separately for rr and grid (graph-level points).