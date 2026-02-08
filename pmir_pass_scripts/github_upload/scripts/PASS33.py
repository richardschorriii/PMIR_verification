#!/usr/bin/env python3
"""
PASS33 - PMIR Verification Test
Extracted from ChatGPT transcript (line 81251)
Length: 8441 characters
"""

#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Returns (coef, yhat, r2)
    """
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, yhat, r2


def bootstrap_se(X: np.ndarray, y: np.ndarray, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 5:
        return np.full(X.shape[1], np.nan)
    coefs = np.zeros((reps, X.shape[1]), dtype=float)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        cr, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        coefs[r] = cr
    return np.std(coefs, axis=0, ddof=1)


def try_statsmodels_se(X: np.ndarray, y: np.ndarray):
    try:
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit()
        return np.asarray(model.bse, dtype=float), float(model.rsquared)
    except Exception:
        return None, None


def build_design(df: pd.DataFrame, with_cross: bool, topo_ref: str):
    """
    Encodes:
      topo dummy = 1 if topology != topo_ref (assumes exactly 2 topologies in df)
      X columns:
        [1, logN, logE, topo_dummy, topo_dummy*logN, topo_dummy*logE] (+ cross terms if requested)
    This gives separate intercept/slope per topology in one model.
    """
    topo_vals = sorted(df["topology"].unique().tolist())
    if len(topo_vals) != 2:
        raise ValueError(f"Expected exactly 2 topologies, got {topo_vals}")
    if topo_ref not in topo_vals:
        topo_ref = topo_vals[0]

    topo_dummy = (df["topology"].to_numpy() != topo_ref).astype(float)
    logN = df["logN"].to_numpy()
    logE = df["logE"].to_numpy()

    cols = []
    X = [np.ones_like(logN), logN, logE, topo_dummy, topo_dummy * logN, topo_dummy * logE]
    cols = ["a", "b_logN", "c_logE", f"d_isNot_{topo_ref}", "e_logN_x_dummy", "f_logE_x_dummy"]

    if with_cross:
        cross = logN * logE
        X += [cross, topo_dummy * cross]
        cols += ["g_logNlogE", "h_logNlogE_x_dummy"]

    X = np.vstack(X).T
    return X, cols, topo_ref, topo_vals


def per_topology_fit(df: pd.DataFrame, with_cross: bool, topo: str):
    s = df[df["topology"] == topo].copy()
    y = s["delta"].to_numpy()
    logN = s["logN"].to_numpy()
    logE = s["logE"].to_numpy()
    if with_cross:
        X = np.vstack([np.ones_like(logN), logN, logE, logN * logE]).T
        cols = ["a", "b_logN", "c_logE", "d_logNlogE"]
    else:
        X = np.vstack([np.ones_like(logN), logN, logE]).T
        cols = ["a", "b_logN", "c_logE"]
    coef, yhat, r2 = ols_fit(X, y)
    se = bootstrap_se(X, y, reps=5000, seed=1337)
    se_sm, r2_sm = try_statsmodels_se(X, y)
    if se_sm is not None:
        se = se_sm
        r2 = r2_sm
    return cols, coef, se, r2, len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="PASS33 contrast table: topology,N,probe_eps,delta_mean_a_minus_b,...")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topo_ref", default="rr", help="reference topology for pooled dummy encoding")
    ap.add_argument("--boot_reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_eps", type=float, default=1e-12)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.in_csv)

    need = {"topology", "N", "probe_eps", "delta_mean_a_minus_b"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing required columns: {miss}")

    # numeric hygiene
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["probe_eps"] = pd.to_numeric(df["probe_eps"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta_mean_a_minus_b"], errors="coerce")
    df = df.dropna(subset=["topology", "N", "probe_eps", "delta"])
    df = df[(df["N"] > 0) & (df["probe_eps"] > args.min_eps)]
    if len(df) < 10:
        raise SystemExit(f"[ERR] too few rows after cleaning: {len(df)}")

    df["logN"] = _safe_log(df["N"].to_numpy(dtype=float))
    df["logE"] = _safe_log(df["probe_eps"].to_numpy(dtype=float))

    # ----- Pooled fits -----
    pooled_rows = []
    for with_cross, label in [(False, "m2_no_cross"), (True, "m3_with_cross")]:
        X, cols, topo_ref, topo_vals = build_design(df, with_cross=with_cross, topo_ref=args.topo_ref)
        y = df["delta"].to_numpy()

        coef, yhat, r2 = ols_fit(X, y)

        se = bootstrap_se(X, y, reps=args.boot_reps, seed=args.seed)
        se_sm, r2_sm = try_statsmodels_se(X, y)
        if se_sm is not None:
            se = se_sm
            r2 = r2_sm

        # unpack into dict
        row = {"block": "pooled_all_eps", "n": int(len(df)), "status": "ok", "model": label, "topo_ref": topo_ref}
        for name, cval, sval in zip(cols, coef, se):
            row[name] = float(cval)
            row[name + "_se"] = float(sval) if np.isfinite(sval) else np.nan
        row["r2"] = float(r2)
        pooled_rows.append(row)

    pooled = pd.DataFrame(pooled_rows)
    pooled_path = outdir / "pass34_pooled_models.csv"
    pooled.to_csv(pooled_path, index=False)

    # ----- Per-topology fits -----
    per_rows = []
    for topo in sorted(df["topology"].unique()):
        for with_cross, label in [(False, "no_cross"), (True, "with_cross")]:
            cols, coef, se, r2, n = per_topology_fit(df, with_cross=with_cross, topo=topo)
            row = {"topology": topo, "model": label, "n": int(n), "r2": float(r2)}
            for name, cval, sval in zip(cols, coef, se):
                row[name] = float(cval)
                row[name + "_se"] = float(sval) if np.isfinite(sval) else np.nan
            per_rows.append(row)

    per = pd.DataFrame(per_rows)
    per_path = outdir / "pass34_per_topology_models.csv"
    per.to_csv(per_path, index=False)

    # ----- Quick text summary -----
    txt = outdir / "pass34_scaling_summary.txt"
    lines = []
    lines.append("PASS34 — scaling regression on PASS33 (probe invariance; metric=delta_mean_a_minus_b)")
    lines.append(f"in_csv: {args.in_csv}")
    lines.append(f"rows used: {len(df)}")
    lines.append(f"topologies: {sorted(df['topology'].unique().tolist())}")
    lines.append("")
    lines.append(f"wrote: {pooled_path}")
    lines.append(f"wrote: {per_path}")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  b_logN  : scaling with N (effective geometry / size dependence)")
    lines.append("  c_logE  : scaling with eps (probe strength response)")
    lines.append("  topo dummy + interactions: topology-dependent metric response (effective curvature difference)")
    lines.append("  logNlogE: whether eps response itself changes with N")
    lines.append("")
    lines.append("POOLED MODELS:")
    for _, r in pooled.iterrows():
        lines.append(f"  {r['model']}: R2={r['r2']:.4f} n={int(r['n'])} topo_ref={r['topo_ref']}")
    txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[PASS34] wrote {pooled_path}")
    print(f"[PASS34] wrote {per_path}")
    print(f"[PASS34] wrote {txt}")


if __name__ == "__main__":
    main()
PowerShell-safe run commands
1) Point at your PASS33 input CSV
This should be the file you generated from your PASS33 run (the table you printed). Example:

$P33 = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass33_probe_invariance_auc\pass33_contrast_by_dir_probe_eps.csv"
$O34 = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass34_scaling_auc"
2) Run PASS34
python .\pass34_scaling_regression_auc.py `
  --in_csv  $P33 `
  --outdir  $O34 `
  --topo_ref rr `
  --boot_reps 5000 `
  --seed 1337
3) Quick print of key outputs
python -c "import pandas as pd; print(pd.read_csv(r'$O34\pass34_pooled_models.csv').to_string(index=False))"
python -c "import pandas as pd; print(pd.read_csv(r'$O34\pass34_per_topology_models.csv').to_string(index=False))"
Outputs created:

pass34_pooled_models.csv

pass34_per_topology_models.csv