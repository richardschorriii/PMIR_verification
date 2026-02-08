#!/usr/bin/env python3
"""
PASS9 - PMIR Verification Test
Extracted from ChatGPT transcript (line 41390)
Length: 10074 characters
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---- small utilities (same as pass8) ----
def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    sum_r1 = ranks[y_true == 1].sum()
    auc = (sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg_l2(X, y, l2=1.0, lr=0.1, steps=4000):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1, dtype=float)
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w

def predict_logreg(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return sigmoid(Xb @ w)

def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def load_features(path: str, mode: str):
    df = pd.read_csv(path)

    # normalize numeric
    for c in ["N","graph_seed","pmir_seed","target_topology"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure logN exists
    if "N" in df.columns:
        df["logN"] = np.log(df["N"].astype(float))

    if mode == "graph":
        key = ["topology","target_topology","N","graph_seed"]
        agg_cols = [
            "lambda2_base","log_inv_l2","logN","ipr_fiedler",
            "tau_half","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"
        ]
        keep = [c for c in (key + agg_cols) if c in df.columns]
        df = df[keep].copy()
        df = df.groupby(key, as_index=False)[[c for c in agg_cols if c in df.columns]].mean()
        df["logN"] = np.log(df["N"].astype(float))

    return df

def eval_cv(df, feat_cols, k=5, seed=0, permute=None):
    """
    permute:
      None -> normal
      'global' -> shuffle y over all rows
      'within_N' -> shuffle y within each N
      'within_graphseed' -> shuffle y within each graph_seed
    """
    d = df.dropna(subset=["target_topology"] + feat_cols).copy()
    if len(d) < 20:
        return dict(n=len(d), acc_mean=np.nan, acc_std=np.nan, auc_mean=np.nan, auc_std=np.nan, folds=[])

    y = d["target_topology"].to_numpy(dtype=int)

    # label permutation (leakage audit)
    if permute == "global":
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)
    elif permute == "within_N":
        rng = np.random.default_rng(seed)
        y2 = y.copy()
        for Nval, idx in d.groupby("N").indices.items():
            idx = np.asarray(list(idx), dtype=int)
            y2[idx] = rng.permutation(y2[idx])
        y = y2
    elif permute == "within_graphseed":
        rng = np.random.default_rng(seed)
        y2 = y.copy()
        for gs, idx in d.groupby("graph_seed").indices.items():
            idx = np.asarray(list(idx), dtype=int)
            y2[idx] = rng.permutation(y2[idx])
        y = y2

    X = d[feat_cols].to_numpy(dtype=float)
    mu, sd = standardize_fit(X)
    Xs = standardize_apply(X, mu, sd)

    folds = kfold_indices(len(d), k=k, seed=seed)
    accs, aucs = [], []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, ytr = Xs[train_idx], y[train_idx]
        Xte, yte = Xs[test_idx], y[test_idx]
        w = train_logreg_l2(Xtr, ytr, l2=1.0, lr=0.1, steps=4000)
        p = predict_logreg(Xte, w)
        yhat = (p >= 0.5).astype(int)
        accs.append(float((yhat == yte).mean()))
        aucs.append(safe_auc(yte, p))

    return dict(
        n=len(d),
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        auc_mean=float(np.nanmean(aucs)),
        auc_std=float(np.nanstd(aucs)),
        folds=[float(x) for x in accs],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["sample","graph"], default="graph")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features(args.features, args.mode)

    base_feats = ["log_inv_l2","logN","ipr_fiedler","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"]
    # keep only what exists (defensive)
    base_feats = [c for c in base_feats if c in df.columns]

    # ---- ablations ----
    rows = []

    # full model
    r_full = eval_cv(df, base_feats, k=args.k, seed=args.seed)
    rows.append({"setting":"FULL", "features":"+".join(base_feats), **r_full})

    # drop-one
    for drop in base_feats:
        feats = [c for c in base_feats if c != drop]
        r = eval_cv(df, feats, k=args.k, seed=args.seed)
        rows.append({"setting":f"DROP_{drop}", "features":"+".join(feats), **r})

    # keep-only sets (as requested)
    keep_sets = {
        "KEEP_struct_only": [c for c in ["t_peak","auc_delta","shape_ratio"] if c in base_feats],
        "KEEP_spec_size_ipr": [c for c in ["log_inv_l2","logN","ipr_fiedler"] if c in base_feats],
        "KEEP_decay_only": [c for c in ["tau_half_lb","censored"] if c in base_feats],
    }
    for nm, feats in keep_sets.items():
        r = eval_cv(df, feats, k=args.k, seed=args.seed)
        rows.append({"setting":nm, "features":"+".join(feats), **r})

    out_csv = outdir / f"pass9_ablation_table_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # short report
    out_txt = outdir / f"pass9_ablation_report_{args.mode}.txt"
    lines = []
    lines.append(f"PASS 9A — Ablation ({args.mode}-level)")
    lines.append(f"Input: {args.features}")
    lines.append(f"Rows (raw after mode transform): {len(df)}")
    lines.append("")
    lines.append(f"[OK] Wrote: {out_csv}")
    lines.append("")
    # show top rows sorted by acc then auc
    tab = pd.DataFrame(rows).sort_values(["acc_mean","auc_mean"], ascending=False)
    show = tab[["setting","n","acc_mean","auc_mean","features"]].head(20)
    lines.append(show.to_string(index=False))
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # ---- permutations ----
    perm_out = outdir / f"pass9_permutation_report_{args.mode}.txt"
    permlines = []
    permlines.append(f"PASS 9B — Permutation sanity ({args.mode}-level)")
    permlines.append(f"Using FULL features: {base_feats}")
    permlines.append("")
    for perm in ["global","within_N","within_graphseed"]:
        r = eval_cv(df, base_feats, k=args.k, seed=args.seed, permute=perm)
        permlines.append(f"{perm}: n={r['n']}  ACC={r['acc_mean']:.4f}±{r['acc_std']:.4f}  AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}  folds={['%.3f'%x for x in r['folds']]}")
    perm_out.write_text("\n".join(permlines), encoding="utf-8")

    print(out_txt.read_text(encoding="utf-8"))
    print("")
    print(perm_out.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
2) pass9_holdoutN.py
Creates:

pass9_holdoutN_report.txt

pass9_holdoutN_table.csv

# pass9_holdoutN.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    sum_r1 = ranks[y_true == 1].sum()
    auc = (sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg_l2(X, y, l2=1.0, lr=0.1, steps=6000):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1, dtype=float)
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w

def predict_logreg(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return sigmoid(Xb @ w)

def load_features(path: str, mode: str):
    df = pd.read_csv(path)
    for c in ["N","graph_seed","pmir_seed","target_topology"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "N" in df.columns:
        df["logN"] = np.log(df["N"].astype(float))
    if mode == "graph":
        key = ["topology","target_topology","N","graph_seed"]
        agg_cols = [
            "lambda2_base","log_inv_l2","logN","ipr_fiedler",
            "tau_half","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"
        ]
        keep = [c for c in (key + agg_cols) if c in df.columns]
        df = df[keep].copy()
        df = df.groupby(key, as_index=False)[[c for c in agg_cols if c in df.columns]].mean()
        df["logN"] = np.log(df["N"].astype(float))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["sample","graph"], default="graph")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)