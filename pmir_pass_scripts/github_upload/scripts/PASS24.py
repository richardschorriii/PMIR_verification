#!/usr/bin/env python3
"""
PASS24 - PMIR Verification Test
Extracted from ChatGPT transcript (line 67931)
Length: 5836 characters
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def pick_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Missing {label}. Have: {list(df.columns)}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="PASS20 by-graph CSV")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--k", type=float, default=1.0, help="ordered criterion: |mean_F| >= k*std_F")
    ap.add_argument("--q", type=float, default=0.90, help="chaotic threshold quantile within (topology,N)")
    ap.add_argument("--prefer_chaotic", action="store_true",
                    help="if set: basin='chaotic' overrides ordered/mixed; else keep separate flag only")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    # expected keys
    for c in ["topology", "N", "graph_seed"]:
        if c not in df.columns:
            raise SystemExit(f"Missing key '{c}'. Have: {list(df.columns)}")

    # tolerate naming variants
    mean_col = pick_col(df, ["mean_F", "mean_F_mean"])
    std_col  = pick_col(df, ["std_F", "std_F_mean"])
    flip_col = pick_col(df, ["sign_flip_rate", "sign_flip_rate_mean"], required=False, label="sign flip column")
    stab_col = pick_col(df, ["stability_ratio", "stability_ratio_mean"], required=False, label="stability ratio column")

    # numeric hygiene
    for c in ["N", "graph_seed", mean_col, std_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if flip_col:
        df[flip_col] = pd.to_numeric(df[flip_col], errors="coerce")
    if stab_col:
        df[stab_col] = pd.to_numeric(df[stab_col], errors="coerce")

    df = df.dropna(subset=["topology", "N", "graph_seed", mean_col, std_col]).copy()

    k = float(args.k)
    q = float(args.q)

    df["abs_mean_F"] = df[mean_col].abs()
    df["ordered_flag"] = df["abs_mean_F"] >= (k * df[std_col])

    # chaotic threshold per (topology,N)
    thr = (
        df.groupby(["topology", "N"])[std_col]
          .quantile(q)
          .reset_index()
          .rename(columns={std_col: "stdF_qthr"})
    )
    df = df.merge(thr, on=["topology", "N"], how="left")
    df["chaotic_flag"] = df[std_col] >= df["stdF_qthr"]

    # base basin (no chaotic override)
    df["basin_base"] = np.where(df["ordered_flag"], "ordered", "mixed")

    # final basin
    if args.prefer_chaotic:
        df["basin"] = np.where(df["chaotic_flag"], "chaotic", df["basin_base"])
    else:
        df["basin"] = df["basin_base"]

    # output columns
    keep = ["topology", "N", "graph_seed", "basin", "basin_base",
            "ordered_flag", "chaotic_flag",
            mean_col, std_col, "abs_mean_F", "stdF_qthr"]
    if flip_col:
        keep.append(flip_col)
    if stab_col:
        keep.append(stab_col)

    out_graph = df[keep].copy()
    out_csv = outdir / "pass23_basins_by_graph.csv"
    out_graph.to_csv(out_csv, index=False)

    # summary by (topology,N)
    def _n(x): return int(x.sum())
    grp = (
        out_graph.groupby(["topology", "N"], as_index=False)
        .agg(
            graphs=("graph_seed", "nunique"),
            n_ordered=("ordered_flag", _n),
            n_chaotic=("chaotic_flag", _n),
            mean_stdF=(std_col, "mean"),
            med_stdF=(std_col, "median"),
            mean_abs_meanF=("abs_mean_F", "mean"),
            med_abs_meanF=("abs_mean_F", "median"),
        )
    )
    grp["frac_ordered"] = grp["n_ordered"] / grp["graphs"].clip(lower=1)
    grp["frac_chaotic"] = grp["n_chaotic"] / grp["graphs"].clip(lower=1)

    out_N = outdir / "pass23_basins_by_N.csv"
    grp.to_csv(out_N, index=False)

    # text report
    lines = []
    lines.append("PASS 23 — Basin classification (graph-level)")
    lines.append(f"in: {Path(args.in_csv)}")
    lines.append(f"rows(in): {len(pd.read_csv(args.in_csv))}")
    lines.append(f"rows(used): {len(out_graph)}")
    lines.append(f"rule ordered: |{mean_col}| >= {args.k}*{std_col}")
    lines.append(f"rule chaotic: {std_col} >= q{int(100*q)}({std_col} | topology,N)")
    lines.append(f"prefer_chaotic_override: {bool(args.prefer_chaotic)}")
    lines.append("")
    lines.append("Summary by topology,N:")
    lines.append(grp.sort_values(["topology", "N"]).to_string(index=False))
    lines.append("")
    lines.append(f"[OK] wrote {out_csv}")
    lines.append(f"[OK] wrote {out_N}")

    out_txt = outdir / "pass23_basins.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
PowerShell run (your current 30-seed pipeline)
$IN20 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass20_seed_stability\pass20_seed_stability_by_graph.csv"
$O23  = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass23_basins"
mkdir $O23 -Force | Out-Null

python .\pass23_basin_classify.py `
  --in_csv $IN20 `
  --outdir $O23 `
  --k 1.0 `
  --q 0.90
If you want “chaotic” to override ordered/mixed in a single basin label:

python .\pass23_basin_classify.py `
  --in_csv $IN20 `
  --outdir $O23 `
  --k 1.0 `
  --q 0.90 `
  --prefer_chaotic
Quick sanity checks
python -c "
import pandas as pd
p=r'.\analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_30seeds\pass23_basins\pass23_basins_by_graph.csv'
d=pd.read_csv(p)
print('rows',len(d))
print('basin counts:\\n', d['basin'].value_counts())
print('chaotic_flag counts:\\n', d['chaotic_flag'].value_counts())
print(d.groupby(['topology','N'])['graph_seed'].nunique())
"
Next after this: we’ll point your existing “effects” pass at ordered vs mixed (and/or chaotic) instead of “vacuum vs control”.