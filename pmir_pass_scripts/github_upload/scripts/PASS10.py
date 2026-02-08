#!/usr/bin/env python3
"""
PASS10 - PMIR Verification Test
Extracted from ChatGPT transcript (line 41851)
Length: 10008 characters
"""

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

def train_logreg_l2(X, y, l2=1.0, lr=0.1, steps=8000):
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

def load_graph_level(features_csv: str):
    df = pd.read_csv(features_csv)

    for c in ["N","graph_seed","pmir_seed","target_topology"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["logN"] = np.log(df["N"].astype(float))

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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / "pass10_logo_graphseed_report.txt"
    out_csv = outdir / "pass10_logo_graphseed_table.csv"

    df = load_graph_level(args.features)

    feat_cols = ["log_inv_l2","logN","ipr_fiedler","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    d = df.dropna(subset=["target_topology","graph_seed"] + feat_cols).copy()

    rows = []
    for gs in sorted(d["graph_seed"].unique().tolist()):
        train = d[d["graph_seed"] != gs].copy()
        test  = d[d["graph_seed"] == gs].copy()

        ytr = train["target_topology"].to_numpy(dtype=int)
        yte = test["target_topology"].to_numpy(dtype=int)

        Xtr = train[feat_cols].to_numpy(dtype=float)
        Xte = test[feat_cols].to_numpy(dtype=float)

        mu, sd = standardize_fit(Xtr)
        Xtr_s = standardize_apply(Xtr, mu, sd)
        Xte_s = standardize_apply(Xte, mu, sd)

        w = train_logreg_l2(Xtr_s, ytr, l2=1.0, lr=0.1, steps=8000)
        p = predict_logreg(Xte_s, w)
        yhat = (p >= 0.5).astype(int)

        acc = float((yhat == yte).mean())
        auc = safe_auc(yte, p)

        rows.append(dict(graph_seed=float(gs), n_train=len(train), n_test=len(test), acc=acc, auc=auc))

    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)

    lines = []
    lines.append("PASS 10A — Leave-one-graphseed-out (graph-level)")
    lines.append(f"Input: {args.features}")
    lines.append(f"Features: {feat_cols}")
    lines.append("")
    lines.append(tab.to_string(index=False))
    lines.append("")
    lines.append(f"ACC mean={tab['acc'].mean():.4f}  std={tab['acc'].std(ddof=0):.4f}")
    lines.append(f"AUC mean={tab['auc'].mean():.4f}  std={tab['auc'].std(ddof=0):.4f}")
    lines.append("")
    lines.append(f"[OK] Wrote: {out_csv}")
    lines.append(f"[OK] Wrote: {out_txt}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
Script 2: pass10_permutation_mc.py
# pass10_permutation_mc.py
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

def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def load_graph_level(features_csv: str):
    df = pd.read_csv(features_csv)
    for c in ["N","graph_seed","pmir_seed","target_topology"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["logN"] = np.log(df["N"].astype(float))

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

def cv_score(d, feat_cols, y_perm=None, k=5, seed=0):
    d = d.dropna(subset=["target_topology"] + feat_cols).copy()
    y = d["target_topology"].to_numpy(dtype=int) if y_perm is None else y_perm
    X = d[feat_cols].to_numpy(dtype=float)

    mu, sd = standardize_fit(X)
    Xs = standardize_apply(X, mu, sd)

    folds = kfold_indices(len(d), k=k, seed=seed)
    accs, aucs = [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w = train_logreg_l2(Xs[tr], y[tr], l2=1.0, lr=0.1, steps=6000)
        p = predict_logreg(Xs[te], w)
        yhat = (p >= 0.5).astype(int)
        accs.append(float((yhat == y[te]).mean()))
        aucs.append(safe_auc(y[te], p))
    return float(np.mean(accs)), float(np.nanmean(aucs))

def permute_labels(d, scheme, rng):
    y = d["target_topology"].to_numpy(dtype=int).copy()
    if scheme == "global":
        return rng.permutation(y)

    y2 = y.copy()
    if scheme == "within_N":
        for Nval, idx in d.groupby("N").indices.items():
            idx = np.asarray(list(idx), dtype=int)
            y2[idx] = rng.permutation(y2[idx])
        return y2

    if scheme == "within_graphseed":
        for gs, idx in d.groupby("graph_seed").indices.items():
            idx = np.asarray(list(idx), dtype=int)
            y2[idx] = rng.permutation(y2[idx])
        return y2

    raise ValueError("unknown scheme")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / "pass10_permutation_mc_report.txt"
    out_csv = outdir / "pass10_permutation_mc_table.csv"

    df = load_graph_level(args.features)
    feat_cols = ["log_inv_l2","logN","ipr_fiedler","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    d = df.dropna(subset=["target_topology"] + feat_cols).copy()

    rng = np.random.default_rng(args.seed)

    # real baseline
    real_acc, real_auc = cv_score(d, feat_cols, y_perm=None, k=args.k, seed=args.seed)

    rows = []
    for scheme in ["global","within_N","within_graphseed"]:
        accs, aucs = [], []
        for r in range(args.reps):
            y_perm = permute_labels(d, scheme, rng)
            acc, auc = cv_score(d, feat_cols, y_perm=y_perm, k=args.k, seed=args.seed + r + 17)
            accs.append(acc); aucs.append(auc)
        rows.append(dict(
            scheme=scheme,
            reps=args.reps,
            acc_mean=float(np.mean(accs)),
            acc_std=float(np.std(accs)),
            acc_p95=float(np.quantile(accs, 0.95)),
            auc_mean=float(np.nanmean(aucs)),
            auc_std=float(np.nanstd(aucs)),
            auc_p95=float(np.nanquantile(aucs, 0.95)),
        ))

    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)

    lines = []
    lines.append("PASS 10B — Permutation Monte Carlo (graph-level)")
    lines.append(f"Input: {args.features}")
    lines.append(f"Features: {feat_cols}")
    lines.append(f"k={args.k}  reps={args.reps}  seed={args.seed}")
    lines.append("")
    lines.append(f"REAL: ACC={real_acc:.4f}  AUC={real_auc:.4f}")
    lines.append("")
    lines.append(tab.to_string(index=False))
    lines.append("")