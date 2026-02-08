#!/usr/bin/env python3
"""
PASS13 - PMIR Verification Test
Extracted from ChatGPT transcript (line 59165)
Length: 9678 characters
"""

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
    w = train_logreg_l2(standardize_apply(Xtr, mu, sd), ytr, l2=1.0, lr=0.1, steps=args.base_steps)

    yte = dte["N_class"].to_numpy(int)
    Xte = dte[FEATS].to_numpy(float)
    Xte0 = standardize_apply(Xte, mu, sd)

    z_base = predict_logit_logreg(Xte0, w)
    auc_base = safe_auc(yte, z_base)

    rng = np.random.default_rng(args.seed)

    rows = []
    for r in range(args.reps):
        # permute each feature independently within test
        for j, f in enumerate(FEATS):
            Xp = Xte0.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, j] = Xp[perm, j]
            z_p = predict_logit_logreg(Xp, w)
            auc_p = safe_auc(yte, z_p)
            rows.append({
                "rep": r,
                "feat": f,
                "auc_base": float(auc_base),
                "auc_perm": float(auc_p),
                "delta_auc": float(auc_base - auc_p),
            })

    tab = pd.DataFrame(rows)
    summ = tab.groupby("feat", as_index=False)["delta_auc"].agg(["mean", "std"]).reset_index()
    summ = summ.rename(columns={"mean": "delta_auc_mean", "std": "delta_auc_std"})
    summ = summ.sort_values("delta_auc_mean", ascending=False)

    out_csv = outdir / f"pass13b_perm_{args.train_topology}_to_{args.test_topology}.csv"
    out_sum = outdir / f"pass13b_perm_{args.train_topology}_to_{args.test_topology}_summary.csv"
    tab.to_csv(out_csv, index=False)
    summ.to_csv(out_sum, index=False)

    lines = []
    lines.append("PASS 13B — Permutation importance on TEST (ΔAUC)")
    lines.append(f"features: {args.features}")
    lines.append(f"train_topology={args.train_topology}  test_topology={args.test_topology}")
    lines.append(f"test n={len(dte)}  reps={args.reps}  seed={args.seed}")
    lines.append(f"AUC base (test) = {auc_base:.4f}")
    lines.append("")
    for _, r in summ.iterrows():
        lines.append(f"- {r['feat']}: ΔAUC={r['delta_auc_mean']:.4f} ± {r['delta_auc_std']:.4f}")

    out_txt = outdir / f"pass13b_perm_{args.train_topology}_to_{args.test_topology}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))

    print(f"\n[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_sum}")
    print(f"[OK] wrote {out_txt}")

if __name__ == "__main__":
    main()
3) pass13c_feature_polarity_audit.py
# pass13c_feature_polarity_audit.py
# PASS 13C — Polarity audit: AUC(y|feat), corr(feat,y) per topology + AUC(y|z) per direction

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

    rows = []

    # Feature-only polarity by topology
    for topo in ["grid2d_periodic", "rr"]:
        d = df[df.topology == topo].dropna(subset=["N_class"] + FEATS).copy()
        y = d["N_class"].to_numpy(int)
        for f in FEATS:
            x = d[f].to_numpy(float)
            rows.append({
                "kind": "feat_only",
                "topology": topo,
                "train_topology": "",
                "test_topology": "",
                "feat": f,
                "auc_y_feat": safe_auc(y, x),
                "corr_feat_y": safe_corr(x, y),
                "auc_y_z": np.nan,
                "corr_z_y": np.nan,
                "w_std": np.nan,
            })

    # Directional model polarity (train->test): does z invert?
    for train_topo, test_topo in [("grid2d_periodic", "rr"), ("rr", "grid2d_periodic")]:
        dtr = df[df.topology == train_topo].dropna(subset=["N_class"] + FEATS).copy()
        dte = df[df.topology == test_topo].dropna(subset=["N_class"] + FEATS).copy()

        ytr = dtr["N_class"].to_numpy(int)
        Xtr = dtr[FEATS].to_numpy(float)
        mu, sd = standardize_fit(Xtr)
        w = train_logreg_l2(standardize_apply(Xtr, mu, sd), ytr, l2=1.0, lr=0.1, steps=args.base_steps)