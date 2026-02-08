#!/usr/bin/env python3
"""
PASS48 - PMIR Verification Test
Extracted from ChatGPT transcript (line 95629)
Length: 4288 characters
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_log(x, floor=1e-300):
    return np.log(np.maximum(x, floor))

def ols_fit(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    return beta, yhat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ycol", default="collapse_metric")
    ap.add_argument("--eps", type=float, default=1e-30)
    ap.add_argument("--predictor", default="gapcv_x_grid")
    ap.add_argument("--controls", nargs="*", default=["logN","probe_eps","topo_is_grid","gap_cv"])
    ap.add_argument("--title", default="Partial residual: logY ⟂ controls vs predictor")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv).copy()

    # Required
    need = [args.ycol, "N", "probe_eps", args.predictor] + args.controls
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing cols: {miss}")

    # logY
    y = safe_log(df[args.ycol].to_numpy(float) + args.eps)

    # Controls matrix
    Xc = [np.ones(len(df))]
    for c in args.controls:
        Xc.append(df[c].to_numpy(float))
    Xc = np.column_stack(Xc)

    # Residualize y wrt controls
    _, yhat_c = ols_fit(Xc, y)
    y_res = y - yhat_c

    # Now plot residual vs predictor
    x = df[args.predictor].to_numpy(float)

    # Fit line for visualization only
    X = np.column_stack([np.ones(len(df)), x])
    b, yhat = ols_fit(X, y_res)

    plt.figure()
    plt.scatter(x, y_res, s=12, alpha=0.5)
    xs = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(xs, b[0] + b[1]*xs)
    plt.xlabel(args.predictor)
    plt.ylabel("residual(logY | controls)")
    plt.title(args.title)
    out = os.path.join(args.outdir, "fig_partial_residual_vs_predictor.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main()
Run it (use the same dataset you used for the interaction model):

python .\pass48_partial_residual_plot.py `
  --csv "$O47\pass47_seedlevel_join_aug.csv" `
  --outdir $O48 `
  --ycol collapse_metric `
  --predictor gapcv_x_grid `
  --controls logN probe_eps topo_is_grid gap_cv `
  --title "Partial residual: log(collapse_metric) ⟂ {logN,probe_eps,topo_is_grid,gap_cv} vs gapcv_x_grid"
2) Topology-stratified scatter + trend lines (rr-only vs grid-only)
Create pass48_stratified_scatter.py:

# pass48_stratified_scatter.py
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_log(x, floor=1e-300):
    return np.log(np.maximum(x, floor))

def fit_line(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ycol", default="collapse_metric")
    ap.add_argument("--xcol", default="gap_cv")
    ap.add_argument("--eps", type=float, default=1e-30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df["logY"] = safe_log(df[args.ycol].to_numpy(float) + args.eps)

    for topo in ["rr", "grid2d_periodic"]:
        d = df[df["topology"] == topo].copy()
        if len(d) == 0:
            continue
        x = d[args.xcol].to_numpy(float)
        y = d["logY"].to_numpy(float)
        b = fit_line(x, y)

        plt.figure()
        plt.scatter(x, y, s=12, alpha=0.5)
        xs = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(xs, b[0] + b[1]*xs)
        plt.xlabel(args.xcol)
        plt.ylabel("log(collapse_metric + 1e-30)")
        plt.title(f"{topo}: logY vs {args.xcol} (seed-level)")
        out = os.path.join(args.outdir, f"fig_stratified_{topo}_logY_vs_{args.xcol}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)

if __name__ == "__main__":
    main()
Run:

python .\pass48_stratified_scatter.py `
  --csv "$O45\pass45_seedlevel_join.csv" `
  --outdir $O48 `
  --ycol collapse_metric `
  --xcol gap_cv
(Using the un-augmented join is fine; it includes topology and gap_cv.)