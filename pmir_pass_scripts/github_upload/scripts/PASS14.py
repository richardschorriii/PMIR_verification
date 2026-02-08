#!/usr/bin/env python3
"""
PASS14 - PMIR Verification Test
Extracted from ChatGPT transcript (line 58982)
Length: 8610 characters
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FEATS = ["feat__tau_half_lb", "feat__t_peak", "feat__auc_delta", "feat__shape_ratio", "feat__censored"]

def sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan

    order = np.argsort(y_score)
    s = y_score[order]
    y = y_true[order]

    ranks = np.empty_like(s, dtype=float)
    i = 0
    r = 1
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i) - 1))
        ranks[i:j] = avg_rank
        r += (j - i)
        i = j

    sum_r1 = float(ranks[y == 1].sum())
    auc = (sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def train_logreg_l2(X, y, l2=1.0, lr=0.1, steps=9000):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1, dtype=float)
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w

def predict_logit_logreg(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return (Xb @ w).astype(float)

def build_labels(df):
    df = df.copy()
    df["logN"] = np.log(df["N"].astype(float))
    df["logN_resid"] = 0.0
    for topo in ["rr", "grid2d_periodic"]:
        m = df["topology"] == topo
        df.loc[m, "logN_resid"] = df.loc[m, "logN"] - df.loc[m, "logN"].mean()
    med = df["logN_resid"].median()
    df["N_class"] = (df["logN_resid"] > med).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["sample"], default="sample")
    ap.add_argument("--train_topology", choices=["grid2d_periodic", "rr"], required=True)
    ap.add_argument("--test_topology", choices=["grid2d_periodic", "rr"], required=True)
    ap.add_argument("--base_steps", type=int, default=9000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df = df[df["topology"].isin(["rr", "grid2d_periodic"])].copy()
    df = build_labels(df)

    for c in ["N_class"] + FEATS:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    dtr = df[df.topology == args.train_topology].dropna(subset=["N_class"] + FEATS).copy()
    dte = df[df.topology == args.test_topology].dropna(subset=["N_class"] + FEATS).copy()

    ytr = dtr["N_class"].to_numpy(int)
    Xtr = dtr[FEATS].to_numpy(float)

    mu, sd = standardize_fit(Xtr)
    Xtr0 = standardize_apply(Xtr, mu, sd)
    w = train_logreg_l2(Xtr0, ytr, l2=1.0, lr=0.1, steps=args.base_steps)

    # weights on standardized features (ignore intercept)
    w0 = w[0]
    ww = w[1:]

    # logits in both domains
    ztr = predict_logit_logreg(Xtr0, w)

    yte = dte["N_class"].to_numpy(int)
    Xte = dte[FEATS].to_numpy(float)
    Xte0 = standardize_apply(Xte, mu, sd)
    zte = predict_logit_logreg(Xte0, w)

    auc_tr = safe_auc(ytr, ztr)
    auc_te = safe_auc(yte, zte)

    # per-feature stats + sign stability
    rows = []
    for j, f in enumerate(FEATS):
        xtr = dtr[f].to_numpy(float)
        xte = dte[f].to_numpy(float)

        rows.append({
            "feat": f,
            "w_std": float(ww[j]),
            "train_mean": float(np.mean(xtr)),
            "train_std": float(np.std(xtr)),
            "test_mean": float(np.mean(xte)),
            "test_std": float(np.std(xte)),
            "corr_feat_y_train": safe_corr(xtr, ytr),
            "corr_feat_y_test": safe_corr(xte, yte),
            "corr_feat_z_train": safe_corr(xtr, ztr),
            "corr_feat_z_test": safe_corr(xte, zte),
        })

    tab = pd.DataFrame(rows)
    tab["w_abs"] = tab["w_std"].abs()
    tab = tab.sort_values("w_abs", ascending=False).drop(columns=["w_abs"])

    out_csv = outdir / f"pass13a_attrib_{args.train_topology}_to_{args.test_topology}.csv"
    tab.to_csv(out_csv, index=False)

    lines = []
    lines.append("PASS 13A — Feature attribution + sign stability")
    lines.append(f"features: {args.features}")
    lines.append(f"train_topology={args.train_topology}  test_topology={args.test_topology}")
    lines.append(f"train n={len(dtr)}  test n={len(dte)}")
    lines.append(f"AUC train={auc_tr:.4f}  AUC test={auc_te:.4f}")
    lines.append(f"intercept={w0:.6f}")
    lines.append("")
    lines.append("Top weights (standardized space):")
    for _, r in tab.iterrows():
        lines.append(
            f"- {r['feat']}: w={r['w_std']:+.4f} | "
            f"corr(feat,y) train={r['corr_feat_y_train']:+.3f} test={r['corr_feat_y_test']:+.3f} | "
            f"corr(feat,z) train={r['corr_feat_z_train']:+.3f} test={r['corr_feat_z_test']:+.3f}"
        )

    out_txt = outdir / f"pass13a_attrib_{args.train_topology}_to_{args.test_topology}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(out_txt.read_text(encoding="utf-8"))
    print(f"\n[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_txt}")

if __name__ == "__main__":
    main()
2) pass13b_permutation_importance_transfer.py
# pass13b_permutation_importance_transfer.py
# PASS 13B — Permutation importance on TEST: ΔAUC per feature

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FEATS = ["feat__tau_half_lb", "feat__t_peak", "feat__auc_delta", "feat__shape_ratio", "feat__censored"]

def sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan

    order = np.argsort(y_score)
    s = y_score[order]
    y = y_true[order]

    ranks = np.empty_like(s, dtype=float)
    i = 0
    r = 1
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i) - 1))
        ranks[i:j] = avg_rank
        r += (j - i)
        i = j

    sum_r1 = float(ranks[y == 1].sum())
    auc = (sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def train_logreg_l2(X, y, l2=1.0, lr=0.1, steps=9000):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1, dtype=float)
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w

def predict_logit_logreg(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return (Xb @ w).astype(float)

def build_labels(df):
    df = df.copy()
    df["logN"] = np.log(df["N"].astype(float))
    df["logN_resid"] = 0.0
    for topo in ["rr", "grid2d_periodic"]:
        m = df["topology"] == topo
        df.loc[m, "logN_resid"] = df.loc[m, "logN"] - df.loc[m, "logN"].mean()
    med = df["logN_resid"].median()
    df["N_class"] = (df["logN_resid"] > med).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["sample"], default="sample")
    ap.add_argument("--train_topology", choices=["grid2d_periodic", "rr"], required=True)
    ap.add_argument("--test_topology", choices=["grid2d_periodic", "rr"], required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_steps", type=int, default=9000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)