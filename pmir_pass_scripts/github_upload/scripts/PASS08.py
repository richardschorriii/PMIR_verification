#!/usr/bin/env python3
"""
PASS8 - PMIR Verification Test
Extracted from ChatGPT transcript (line 47269)
Length: 10496 characters
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FEATS = ["log_inv_l2","logN","ipr_fiedler","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"]

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    sum_r1 = float(ranks[y_true == 1].sum())
    return float((sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sigmoid(z):
    # keep this wide; clipping (if desired) is done explicitly on logits elsewhere
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

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

def predict_logreg(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return sigmoid(Xb @ w)

def predict_logit(X, w):
    # raw linear score before sigmoid
    Xb = np.c_[np.ones(len(X)), X]
    return (Xb @ w).astype(float)

def best_threshold_grid(y, p):
    # fixed threshold grid to reduce tiny-sample overfit
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    ts = np.linspace(0.0, 1.0, 501)
    best_t = 0.5
    best_bal = -1.0
    for t in ts:
        yhat = (p >= t).astype(int)
        tp = ((yhat == 1) & (y == 1)).sum()
        tn = ((yhat == 0) & (y == 0)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        fn = ((yhat == 0) & (y == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal = 0.5 * (tpr + tnr)
        if bal > best_bal:
            best_bal = float(bal)
            best_t = float(t)
    return best_t, best_bal

def build_graph_level(df):
    for c in ["N","graph_seed","pmir_seed","target_topology"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["logN"] = np.log(df["N"].astype(float))

    key = ["topology","target_topology","N","graph_seed"]
    agg_cols = ["lambda2_base","log_inv_l2","logN","ipr_fiedler",
                "tau_half","tau_half_lb","t_peak","auc_delta","shape_ratio","censored"]
    keep = [c for c in (key + agg_cols) if c in df.columns]
    d = df[keep].copy()
    d = d.groupby(key, as_index=False)[[c for c in agg_cols if c in d.columns]].mean()
    d["logN"] = np.log(d["N"].astype(float))
    return d

def confusion_counts(yhat, y):
    yhat = np.asarray(yhat).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, tn, fp, fn

def eval_transfer(dfA, dfB, feat_cols, ycol, logit_clip=8.0):
    A = dfA.dropna(subset=[ycol] + feat_cols).copy()
    B = dfB.dropna(subset=[ycol] + feat_cols).copy()

    if len(A) < 8 or len(B) < 8:
        return {"n_train": len(A), "n_test": len(B), "status": "insufficient"}

    if A[ycol].nunique() < 2:
        return {"n_train": len(A), "n_test": len(B), "status": "train_single_class"}

    yA = A[ycol].to_numpy(dtype=int)
    XA = A[feat_cols].to_numpy(dtype=float)
    yB = B[ycol].to_numpy(dtype=int)
    XB = B[feat_cols].to_numpy(dtype=float)

    mu, sd = standardize_fit(XA)
    XAs = standardize_apply(XA, mu, sd)
    XBs = standardize_apply(XB, mu, sd)

    w = train_logreg_l2(XAs, yA, l2=1.0, lr=0.1, steps=9000)

    # TRAIN probs (for threshold selection)
    pA = predict_logreg(XAs, w)
    t_star, bal_train = best_threshold_grid(yA, pA)

    # TEST logits + probs (raw + clipped)
    zB = predict_logit(XBs, w)
    pB = sigmoid(zB)

    if logit_clip is not None and logit_clip > 0:
        zB_clip = np.clip(zB, -float(logit_clip), float(logit_clip))
        pB_clip = sigmoid(zB_clip)
    else:
        zB_clip = zB.copy()
        pB_clip = pB.copy()

    # predictions using TRAIN-derived threshold
    yhat_star = (pB_clip >= t_star).astype(int)
    acc_star = float((yhat_star == yB).mean())
    auc = safe_auc(yB, pB_clip)

    # baseline 0.5 threshold
    yhat_05 = (pB_clip >= 0.5).astype(int)
    acc_05 = float((yhat_05 == yB).mean())

    tp, tn, fp, fn = confusion_counts(yhat_star, yB)

    # calibration / diagnostics
    brier = float(np.mean((pB_clip - yB) ** 2))
    p_pos_mean = float(np.mean(pB_clip[yB == 1])) if np.any(yB == 1) else np.nan
    p_neg_mean = float(np.mean(pB_clip[yB == 0])) if np.any(yB == 0) else np.nan

    # logit diagnostics (raw, unclipped)
    z_pos_mean = float(np.mean(zB[yB == 1])) if np.any(yB == 1) else np.nan
    z_neg_mean = float(np.mean(zB[yB == 0])) if np.any(yB == 0) else np.nan
    z_abs_p95  = float(np.percentile(np.abs(zB), 95)) if len(zB) else np.nan
    sat_hi = float(np.mean(pB > 0.999))
    sat_lo = float(np.mean(pB < 0.001))

    return {
        "n_train": len(A),
        "n_test": len(B),
        "acc": acc_star,
        "auc": auc,
        "acc_05": acc_05,
        "brier": brier,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "t_star": float(t_star),
        "bal_train": float(bal_train),
        "p_pos_mean": p_pos_mean,
        "p_neg_mean": p_neg_mean,
        "z_pos_mean": z_pos_mean,
        "z_neg_mean": z_neg_mean,
        "z_abs_p95": z_abs_p95,
        "sat_hi": sat_hi,
        "sat_lo": sat_lo,
        "logit_clip": float(logit_clip) if (logit_clip is not None and logit_clip > 0) else np.nan,
        "status": "ok",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["sample","graph"], default="graph")
    ap.add_argument("--median_split", action="store_true")
    ap.add_argument("--logit_clip", type=float, default=8.0,
                    help="Clip test logits to [-logit_clip, +logit_clip] before sigmoid. Set <=0 to disable.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "pass11e_logitclip_transfer_table.csv"
    out_txt = outdir / "pass11e_logitclip_transfer_report.txt"

    df = pd.read_csv(args.features)

    if args.mode == "graph":
        df = build_graph_level(df)
    else:
        for c in ["N","graph_seed","pmir_seed","target_topology"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["logN"] = np.log(df["N"].astype(float))

    df = df[df["topology"].isin(["grid2d_periodic","rr"])].copy()

    # define N_class
    if args.median_split:
        df["N_class"] = 0
        for topo in ["grid2d_periodic","rr"]:
            m = df["topology"] == topo
            med = df.loc[m, "N"].median()
            df.loc[m, "N_class"] = (df.loc[m, "N"] > med).astype(int)
        split_label = "within-topology median"
    else:
        med = df["N"].median()
        df["N_class"] = (df["N"] > med).astype(int)
        split_label = "global median"

    # ensure features exist
    for c in FEATS:
        if c not in df.columns:
            raise SystemExit(f"Missing feature column: {c}")

    grid = df[df["topology"] == "grid2d_periodic"].copy()
    rr   = df[df["topology"] == "rr"].copy()

    r1 = eval_transfer(grid, rr, FEATS, "N_class", logit_clip=args.logit_clip)
    r2 = eval_transfer(rr, grid, FEATS, "N_class", logit_clip=args.logit_clip)

    tab = pd.DataFrame([
        {"train_topology": "grid2d_periodic", "test_topology": "rr", **r1},
        {"train_topology": "rr", "test_topology": "grid2d_periodic", **r2},
    ])
    tab.to_csv(out_csv, index=False)

    lines = []
    lines.append("PASS 11E — Cross-topology transfer: logit diagnostics + optional logit-clip")
    lines.append(f"Input: {args.features}")
    lines.append(f"Mode: {args.mode}  rows={len(df)}")
    lines.append(f"N split: {split_label}")
    lines.append(f"Features: {FEATS}")
    lines.append(f"logit_clip: {args.logit_clip}")
    lines.append("")
    for _, row in tab.iterrows():
        lines.append(
            f"{row['train_topology']}→{row['test_topology']}: "
            f"n_train={int(row['n_train'])} n_test={int(row['n_test'])} "
            f"acc={row.get('acc', np.nan)} acc@0.5={row.get('acc_05', np.nan)} auc={row.get('auc', np.nan)} "
            f"brier={row.get('brier', np.nan)} "
            f"tp={row.get('tp', np.nan)} tn={row.get('tn', np.nan)} fp={row.get('fp', np.nan)} fn={row.get('fn', np.nan)} "
            f"t*={row.get('t_star', np.nan)} bal_train={row.get('bal_train', np.nan)} "
            f"p_pos_mean={row.get('p_pos_mean', np.nan)} p_neg_mean={row.get('p_neg_mean', np.nan)} "
            f"z_pos_mean={row.get('z_pos_mean', np.nan)} z_neg_mean={row.get('z_neg_mean', np.nan)} "
            f"z_abs_p95={row.get('z_abs_p95', np.nan)} sat_hi={row.get('sat_hi', np.nan)} sat_lo={row.get('sat_lo', np.nan)} "
            f"status={row.get('status', '')}"
        )
    lines.append("")
    lines.append(f"[OK] Wrote: {out_csv}")
    lines.append(f"[OK] Wrote: {out_txt}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
Run it twice to confirm “saturation” is the culprit:

# baseline (no clipping)
python .\pass11e_logit_clip_transfer.py `
  --features .\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass8_classifier\pass8_features.csv `
  --outdir   .\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass11_classifier_checks `
  --mode graph --median_split --logit_clip -1

# with clipping
python .\pass11e_logit_clip_transfer.py `
  --features .\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass8_classifier\pass8_features.csv `
  --outdir   .\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass11_classifier_checks `
  --mode graph --median_split --logit_clip 8
If rr→grid improves when clipping is on and sat_hi drops from ~1.0, you’ve nailed it as logit/probability saturation (calibration shift), not a sign flip.