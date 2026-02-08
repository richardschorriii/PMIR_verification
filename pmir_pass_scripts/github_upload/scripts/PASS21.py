#!/usr/bin/env python3
"""
PASS21 - PMIR Verification Test
Extracted from ChatGPT transcript (line 63561)
Length: 8589 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vacua_csv", required=True, help="PASS22 pass22_pmir_vacua.csv")
    ap.add_argument("--stability_csv", required=True, help="PASS21 pass21_seed_stability_by_graph.csv")
    ap.add_argument("--pmir_field_csv", required=True, help="PASS19 pass19_pmir_field_index.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=3, help="controls per vacuum")
    ap.add_argument("--match_on", choices=["mean_abs_F", "stability_ratio"], default="mean_abs_F")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vac = pd.read_csv(args.vacua_csv)
    stab = pd.read_csv(args.stability_csv)
    fld = pd.read_csv(args.pmir_field_csv)

    # canonicalize needed cols
    for df in [vac, stab, fld]:
        for c in ["N", "graph_seed", "pmir_seed"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute stability_ratio if missing in stab
    if "stability_ratio" not in stab.columns:
        if ("mean_abs_F" in stab.columns) and ("std_F" in stab.columns):
            stab["mean_abs_F"] = pd.to_numeric(stab["mean_abs_F"], errors="coerce")
            stab["std_F"] = pd.to_numeric(stab["std_F"], errors="coerce")
            stab["stability_ratio"] = stab["mean_abs_F"] / (stab["std_F"] + 1e-12)
        else:
            raise SystemExit(f"stability_ratio missing and cannot compute. Have: {list(stab.columns)}")

    # merge vacuum_flag onto stability table
    m = stab.merge(
        vac[["topology", "N", "graph_seed", "vacuum_flag"]],
        on=["topology", "N", "graph_seed"],
        how="left",
    )
    m["vacuum_flag"] = m["vacuum_flag"].fillna(False).astype(bool)

    # choose match metric
    metric = args.match_on
    if metric not in m.columns:
        raise SystemExit(f"match_on={metric} not in merged table. Have: {list(m.columns)}")

    vac_rows = m[m["vacuum_flag"]].copy()
    if len(vac_rows) == 0:
        raise SystemExit("No vacuum rows found in merged table.")

    pair_rows = []
    panel_rows = []

    # helper: attach PMIR_F rows for a given (topology,N,graph_seed) across pmir_seed
    def add_panel(label, topo, N, gseed, role):
        sub = fld[(fld["topology"] == topo) & (fld["N"] == N) & (fld["graph_seed"] == gseed)].copy()
        if len(sub) == 0:
            return
        sub["role"] = role  # vacuum / control
        sub["set_label"] = label
        panel_rows.append(sub)

    for _, v in vac_rows.iterrows():
        topo = v["topology"]
        N = int(v["N"])
        g_v = int(v["graph_seed"])
        v_metric = float(v[metric])

        pool = m[(m["topology"] == topo) & (m["N"] == N) & (~m["vacuum_flag"])].copy()
        if len(pool) == 0:
            continue

        pool["dist"] = (pool[metric].astype(float) - v_metric).abs()
        pool = pool.sort_values(["dist", "graph_seed"]).head(max(args.k * 3, args.k))

        # if ties or small pool, sample without replacement from top window
        choose_from = pool.head(min(len(pool), max(args.k * 2, args.k))).copy()
        if len(choose_from) < args.k:
            chosen = choose_from
        else:
            chosen = choose_from.sample(args.k, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))

        label = f"{topo}_N{N}_vacG{g_v}"
        add_panel(label, topo, N, g_v, "vacuum")

        for _, c in chosen.iterrows():
            g_c = int(c["graph_seed"])
            pair_rows.append({
                "set_label": label,
                "topology": topo,
                "N": N,
                "vac_graph_seed": g_v,
                "ctrl_graph_seed": g_c,
                "metric": metric,
                "vac_metric": v_metric,
                "ctrl_metric": float(c[metric]),
                "abs_dist": float(abs(float(c[metric]) - v_metric)),
            })
            add_panel(label, topo, N, g_c, "control")

    pairs = pd.DataFrame(pair_rows)
    if len(pairs) == 0:
        raise SystemExit("No control matches produced (pool empty for all vacua?).")

    panel = pd.concat(panel_rows, ignore_index=True)
    # summarize at graph level within each set_label/role
    gsum = panel.groupby(["set_label", "role", "topology", "N", "graph_seed"], as_index=False).agg(
        pmir_seeds=("pmir_seed", "nunique"),
        mean_P=("PMIR_F", "mean"),
        std_P=("PMIR_F", "std"),
        mean_absP=("abs_z0", "mean"),
    )

    out_pairs = outdir / "pass23_matched_pairs.csv"
    out_panel = outdir / "pass23_matched_panels.csv"
    out_sum = outdir / "pass23_matched_summary.csv"
    pairs.to_csv(out_pairs, index=False)
    panel.to_csv(out_panel, index=False)
    gsum.to_csv(out_sum, index=False)

    print("PASS 23 — matched controls + panels")
    print("vacua sets:", pairs["set_label"].nunique())
    print("[OK] wrote", out_pairs)
    print("[OK] wrote", out_panel)
    print("[OK] wrote", out_sum)


if __name__ == "__main__":
    main()
Run Pass 23 (PowerShell)
$V22 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass22_vacua\pass22_pmir_vacua.csv"
$S21 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass21_seed_stability\pass21_seed_stability_by_graph.csv"
$F19 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass19_pmir_field\pass19_pmir_field_index.csv"
$O23 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass23_matched"

python .\pass23_match_controls_and_panelize.py `
  --vacua_csv $V22 `
  --stability_csv $S21 `
  --pmir_field_csv $F19 `
  --outdir $O23 `
  --k 4 --match_on mean_abs_F --seed 0
Pass 24 — Vacuum vs control effect size + AUROC
Goal: show that your vacuum is a distinct “phase” in PMIR_F distribution.

Create pass24_vacuum_effects.py:

# pass24_vacuum_effects.py
# PASS 24 — Vacuum vs control effect sizes on PMIR_F (and abs_z0)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s)
    s = s[order]; y = y[order]
    ranks = np.empty_like(s, dtype=float)
    i = 0; r = 1; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg = 0.5 * (r + (r + (j - i) - 1))
        ranks[i:j] = avg
        r += (j - i)
        i = j
    sum_r1 = float(ranks[y == 1].sum())
    return float((sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cohens_d(x1, x0):
    x1 = np.asarray(x1, float); x0 = np.asarray(x0, float)
    m1, m0 = np.mean(x1), np.mean(x0)
    s1, s0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    n1, n0 = len(x1), len(x0)
    sp = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / max(n1 + n0 - 2, 1))
    if sp < 1e-12:
        return np.nan
    return float((m1 - m0) / sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_csv", required=True, help="PASS23 pass23_matched_panels.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.panels_csv)

    # vacuum=1, control=0
    y = (d["role"] == "vacuum").astype(int).to_numpy()

    rows = []
    for score_col in ["PMIR_F", "abs_z0", "z0"]:
        s = d[score_col].to_numpy(float)
        x1 = s[y == 1]
        x0 = s[y == 0]
        if len(x1) == 0 or len(x0) == 0:
            continue
        rows.append({
            "score": score_col,
            "n_vac": int(len(x1)),
            "n_ctrl": int(len(x0)),
            "mean_vac": float(np.mean(x1)),
            "mean_ctrl": float(np.mean(x0)),
            "delta_mean": float(np.mean(x1) - np.mean(x0)),
            "d_cohen": cohens_d(x1, x0),
            "auroc": safe_auc(y, s),
        })

    out = pd.DataFrame(rows)
    out_csv = outdir / "pass24_vacuum_effects.csv"
    out.to_csv(out_csv, index=False)

    txt = outdir / "pass24_vacuum_effects.txt"
    txt.write_text(out.to_string(index=False), encoding="utf-8")
    print(txt.read_text(encoding="utf-8"))
    print("[OK] wrote", out_csv)
    print("[OK] wrote", txt)


if __name__ == "__main__":
    main()
Run:

$P23 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass23_matched\pass23_matched_panels.csv"
$O24 = ".\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass24_vacuum_effects"