#!/usr/bin/env python3
"""
PASS3 - PMIR Verification Test
Extracted from ChatGPT transcript (line 37519)
Length: 6503 characters
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fit_loglog(N, y):
    """
    log(y) = c + a*log(N)
    returns dict with a, c, se_a, r2, n
    """
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(N) & np.isfinite(y) & (N > 0) & (y > 0)
    N = N[m]
    y = y[m]
    n = len(y)
    if n < 3:
        return {"n": n, "a": np.nan, "c": np.nan, "se_a": np.nan, "r2": np.nan}

    x = np.log(N)
    z = np.log(y)
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    c, a = beta[0], beta[1]
    zhat = X @ beta

    # r2 in log space
    ss_res = np.sum((z - zhat) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # standard error of slope
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_a = math.sqrt(s2 * xtx_inv[1, 1]) if np.isfinite(s2) else np.nan

    return {"n": n, "a": float(a), "c": float(c), "se_a": float(se_a), "r2": float(r2)}


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3:
        return np.nan
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    if den <= 0:
        return np.nan
    return float(np.sum(aa * bb) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--pass_id", type=str, default="P0_BASELINE")
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df = _to_num(df, ["N", "probe_eps", "tau_half", "t_peak", "auc_delta", "lambda2_base", "censored"])

    # filter to pass + eps
    d = df.copy()
    if "pass_id" in d.columns:
        d = d[d["pass_id"] == args.pass_id]
    d = d[np.isfinite(d["probe_eps"]) & (np.abs(d["probe_eps"] - args.eps) < 1e-12)]
    d = d.dropna(subset=["topology", "N"])

    # metrics
    metrics = [
        ("t_peak", "ALL", "t_peak"),
        ("auc_delta", "ALL", "auc_delta"),
        ("tau_half", "UNCENSORED", "tau_half"),
    ]

    lines = []
    lines.append("PASS 4 — SCALING LAW (N-dependence)")
    lines.append(f"Input: {in_csv}")
    lines.append(f"Filter: pass_id={args.pass_id}  eps={args.eps}")
    lines.append(f"Rows after filter: {len(d)}")
    lines.append("")

    table_rows = []

    for metric, mode, col in metrics:
        lines.append(f"== {metric} scaling ==")
        for topo, sub0 in d.groupby("topology"):
            sub = sub0.copy()

            if mode == "UNCENSORED":
                if "censored" in sub.columns:
                    sub = sub[(sub["censored"] == 0)]
                sub = sub.dropna(subset=[col])
            else:
                sub = sub.dropna(subset=[col])

            # require positive for loglog fit
            sub = sub[np.isfinite(sub[col]) & (sub[col] > 0)]
            sub = sub[np.isfinite(sub["N"]) & (sub["N"] > 0)]

            fit = fit_loglog(sub["N"].to_numpy(), sub[col].to_numpy())

            # residual sanity vs log(1/lambda2) if lambda2 exists
            resid_corr = np.nan
            if "lambda2_base" in sub.columns and np.isfinite(sub["lambda2_base"]).any():
                lam = sub["lambda2_base"].to_numpy(dtype=float)
                ok = np.isfinite(lam) & (lam > 0)
                if np.count_nonzero(ok) >= 3 and np.isfinite(fit["a"]) and np.isfinite(fit["c"]):
                    # predicted log(y)
                    x = np.log(sub["N"].to_numpy(dtype=float))
                    z = np.log(sub[col].to_numpy(dtype=float))
                    zhat = fit["c"] + fit["a"] * x
                    r = z - zhat
                    resid_corr = corr(r[ok], np.log(1.0 / lam[ok]))

            lines.append(
                f"{topo}: n={fit['n']}  alpha={fit['a']:.6f}  se={fit['se_a']:.6f}  R2(log)={fit['r2']:.4f}  resid_corr_vs_log1/l2={resid_corr:.4f}"
            )

            table_rows.append({
                "metric": metric,
                "mode": mode,
                "topology": topo,
                "n": fit["n"],
                "alpha": fit["a"],
                "se_alpha": fit["se_a"],
                "r2_log": fit["r2"],
                "resid_corr_vs_log1_over_lambda2": resid_corr,
            })

        lines.append("")

        # compare slopes grid vs rr if both present
        t = pd.DataFrame([r for r in table_rows if r["metric"] == metric])
        if set(t["topology"].unique()) >= {"grid2d_periodic", "rr"}:
            g = t[t["topology"] == "grid2d_periodic"].iloc[-1]
            r = t[t["topology"] == "rr"].iloc[-1]
            if np.isfinite(g["alpha"]) and np.isfinite(r["alpha"]) and np.isfinite(g["se_alpha"]) and np.isfinite(r["se_alpha"]):
                delta = float(g["alpha"] - r["alpha"])
                se = float(np.sqrt(g["se_alpha"] ** 2 + r["se_alpha"] ** 2))
                z = delta / se if se > 0 else np.nan
                lines.append(f"Slope diff (grid - rr): Δalpha={delta:.6f}  SE≈{se:.6f}  z≈{z:.3f}")
                lines.append("")
            else:
                lines.append("Slope diff (grid - rr): insufficient finite fits")
                lines.append("")

    out_txt = out_dir / "pass4_scaling_stats.txt"
    out_tbl = out_dir / "pass4_scaling_table.csv"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(out_tbl, index=False)

    print("[OK] Wrote:", str(out_txt))
    print("[OK] Wrote:", str(out_tbl))


if __name__ == "__main__":
    main()
2) Run it (PowerShell safe)
Copy/paste exactly:

python .\pmir_pass4_scaling.py `
  --in_csv .\analysis_transfer\PMIR_TIME_LOCK_MASTER\tauhalf_master_MERGED_tmax48_plus_tmax96.csv `
  --out_dir .\analysis_transfer\PMIR_TIME_LOCK_MASTER\pass4_scaling `
  --pass_id P0_BASELINE `
  --eps 0.05
3) What you should see
Two files created:

analysis_transfer\PMIR_TIME_LOCK_MASTER\pass4_scaling\pass4_scaling_stats.txt

analysis_transfer\PMIR_TIME_LOCK_MASTER\pass4_scaling\pass4_scaling_table.csv