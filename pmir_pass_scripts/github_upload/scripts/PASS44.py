#!/usr/bin/env python3
"""
PASS44 - PMIR Verification Test
Extracted from ChatGPT transcript (line 91886)
Length: 9204 characters
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_seed_from_filename(path: str):
    # expects ..._g{seed}_k{K}.npz in file, but we already have 'seed' column in pass42.
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)          # pass36_pairs_topoN_eps.csv
    ap.add_argument("--pass42_csv", required=True)         # pass42_eigs_band_metrics.csv
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topo_ref", default="rr")
    ap.add_argument("--topo_alt", default="grid2d_periodic")
    ap.add_argument("--use_abs", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.outdir)
    ensure_dir(out)

    pairs = pd.read_csv(args.pairs_csv)
    spec  = pd.read_csv(args.pass42_csv)

    # Basic cleaning
    for c in ["N","seed_ref","seed_alt","probe_eps"]:
        if c in pairs.columns:
            pairs[c] = pd.to_numeric(pairs[c], errors="coerce")
    pairs = pairs.dropna(subset=["N","probe_eps"])
    pairs["N"] = pairs["N"].astype(int)

    # Require topology columns in pairs (if not present, assume fixed ref/alt from args)
    if "topology_ref" not in pairs.columns:
        pairs["topology_ref"] = args.topo_ref
    if "topology_alt" not in pairs.columns:
        pairs["topology_alt"] = args.topo_alt

    # If seeds missing in pairs, fail with actionable message
    if "seed_ref" not in pairs.columns or "seed_alt" not in pairs.columns:
        raise SystemExit("[ERR] pairs_csv must include seed_ref and seed_alt columns for seed-level join.")

    pairs["seed_ref"] = pairs["seed_ref"].astype(int)
    pairs["seed_alt"] = pairs["seed_alt"].astype(int)

    # Filter to kfile=48 in spectral (since your cache is k48)
    if "kfile" in spec.columns:
        spec = spec[spec["kfile"] == 48].copy()

    # Join spectral for ref
    ref = spec.rename(columns={
        "topology":"topology_ref",
        "seed":"seed_ref",
        "gap_cv":"gap_cv_ref",
        "gap_mean":"gap_mean_ref",
        "gap_std":"gap_std_ref",
    })[["topology_ref","N","seed_ref","gap_cv_ref","gap_mean_ref","gap_std_ref"]]

    # Join spectral for alt
    alt = spec.rename(columns={
        "topology":"topology_alt",
        "seed":"seed_alt",
        "gap_cv":"gap_cv_alt",
        "gap_mean":"gap_mean_alt",
        "gap_std":"gap_std_alt",
    })[["topology_alt","N","seed_alt","gap_cv_alt","gap_mean_alt","gap_std_alt"]]

    df = pairs.merge(ref, on=["topology_ref","N","seed_ref"], how="left")
    df = df.merge(alt, on=["topology_alt","N","seed_alt"], how="left")

    # spectral deltas
    df["d_gap_cv"]   = df["gap_cv_alt"]   - df["gap_cv_ref"]
    df["d_gap_mean"] = df["gap_mean_alt"] - df["gap_mean_ref"]
    df["d_gap_std"]  = df["gap_std_alt"]  - df["gap_std_ref"]

    # ratio/logR
    if args.use_abs and "ratio" in df.columns:
        df["ratio"] = df["ratio"].abs()
    if "ratio" in df.columns:
        df["logR"] = np.log(np.maximum(df["ratio"].values.astype(float), 1e-300))
    df["logN"] = np.log(df["N"].values.astype(float))

    # Drop rows where spectral not found
    before = len(df)
    df = df.dropna(subset=["gap_cv_ref","gap_cv_alt","d_gap_cv"])
    dropped = before - len(df)

    df.to_csv(out / "pass44_pairs_with_seed_spectral.csv", index=False)

    (out / "pass44_seedjoin_summary.txt").write_text(
        "PASS44 (revised) — seed-level spectral join\n"
        f"pairs_csv: {args.pairs_csv}\n"
        f"pass42_csv: {args.pass42_csv}\n"
        f"topo_ref: {args.topo_ref}  topo_alt: {args.topo_alt}\n"
        f"rows_out: {len(df)}  rows_dropped_missing_spectral: {dropped}\n",
        encoding="utf-8"
    )

    print("[PASS44] wrote", str(out / "pass44_pairs_with_seed_spectral.csv"))
    print("[PASS44] wrote", str(out / "pass44_seedjoin_summary.txt"))


if __name__ == "__main__":
    main()
Run it
$O44 = "analysis_transfer\PMIR_TIME_LOCK_MASTER\runs_eps_sweep\probe_invariance\pass44_pooled_spectral_seed"
python .\pass44_seedlevel_spectral_join.py --pairs_csv $P36_PAIRS --pass42_csv $O42\pass42_eigs_band_metrics.csv --outdir $O44 --topo_ref rr --topo_alt grid2d_periodic --use_abs 1
python -c "print(open(r'$O44\pass44_seedjoin_summary.txt','r',encoding='utf-8',errors='replace').read())"
2) Pooled regression test using seed-level d_gap_cv
Save as: pass44_seedlevel_pooled_regression.py

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ols_r2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ssr / sst if sst > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_with_seed_spectral_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--boot_reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--predictor", default="d_gap_cv")  # try d_gap_cv, d_gap_mean, d_gap_std
    args = ap.parse_args()

    out = Path(args.outdir)
    ensure_dir(out)

    df = pd.read_csv(args.pairs_with_seed_spectral_csv)

    need = ["probe_dir_mode","probe_mode","probe_eps","N","logR","logN",args.predictor]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing cols: {miss}")

    df = df.dropna(subset=["logR","logN",args.predictor])
    y = df["logR"].values.astype(float)

    # baseline: logR ~ 1 + logN
    X0 = np.column_stack([np.ones(len(df)), df["logN"].values.astype(float)])
    r2_0 = ols_r2(X0, y)

    # augmented: + predictor
    xpred = df[args.predictor].values.astype(float)
    X1 = np.column_stack([np.ones(len(df)), df["logN"].values.astype(float), xpred])
    r2_1 = ols_r2(X1, y)

    delta_r2 = r2_1 - r2_0

    # bootstrap by resampling probe groups (not individual rows)
    rng = np.random.default_rng(args.seed)
    gcols = ["probe_dir_mode","probe_mode","probe_eps"]
    groups = list(df.groupby(gcols).groups.keys())

    dr2 = []
    for _ in range(args.boot_reps):
        sample = rng.choice(len(groups), size=len(groups), replace=True)
        idx = []
        for j in sample:
            idx.extend(df.groupby(gcols).get_group(groups[j]).index.tolist())
        sub = df.loc[idx]
        yb = sub["logR"].values.astype(float)
        X0b = np.column_stack([np.ones(len(sub)), sub["logN"].values.astype(float)])
        X1b = np.column_stack([np.ones(len(sub)), sub["logN"].values.astype(float), sub[args.predictor].values.astype(float)])
        r0 = ols_r2(X0b, yb)
        r1 = ols_r2(X1b, yb)
        if np.isfinite(r0) and np.isfinite(r1):
            dr2.append(r1 - r0)

    dr2 = np.asarray(dr2, dtype=float)
    ci_lo = float(np.quantile(dr2, 0.025)) if len(dr2) else np.nan
    ci_hi = float(np.quantile(dr2, 0.975)) if len(dr2) else np.nan

    pd.DataFrame([{
        "predictor": args.predictor,
        "r2_baseline_logN": r2_0,
        "r2_aug": r2_1,
        "delta_r2": delta_r2,
        "delta_r2_ci_lo": ci_lo,
        "delta_r2_ci_hi": ci_hi,
        "n_rows": len(df),
        "n_groups": len(groups),
        "N_unique": int(df["N"].nunique()),
        "boot_reps": args.boot_reps
    }]).to_csv(out / "pass44_seedlevel_results.csv", index=False)

    (out / "pass44_seedlevel_summary.txt").write_text(
        "PASS44 — pooled regression with seed-level spectral predictor\n"
        f"in: {args.pairs_with_seed_spectral_csv}\n"
        f"predictor: {args.predictor}\n"
        f"N_unique: {df['N'].nunique()}  groups: {len(groups)}  rows: {len(df)}\n\n"
        f"R2 baseline (logR ~ 1 + logN): {r2_0:.6f}\n"
        f"R2 augmented ( + {args.predictor} ): {r2_1:.6f}\n"
        f"ΔR2: {delta_r2:.6f}\n"
        f"Bootstrap ΔR2 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}] (n={len(dr2)})\n",
        encoding="utf-8"
    )

    print("[PASS44] wrote", str(out / "pass44_seedlevel_results.csv"))
    print("[PASS44] wrote", str(out / "pass44_seedlevel_summary.txt"))


if __name__ == "__main__":
    main()
Run it (try multiple predictors)
python .\pass44_seedlevel_pooled_regression.py --pairs_with_seed_spectral_csv $O44\pass44_pairs_with_seed_spectral.csv --outdir $O44 --predictor d_gap_cv --boot_reps 5000 --seed 1337
python -c "print(open(r'$O44\pass44_seedlevel_summary.txt','r',encoding='utf-8',errors='replace').read())"
python -c "import pandas as pd; print(pd.read_csv(r'$O44\pass44_seedlevel_results.csv').to_string(index=False))"

# optional
python .\pass44_seedlevel_pooled_regression.py --pairs_with_seed_spectral_csv $O44\pass44_pairs_with_seed_spectral.csv --outdir $O44 --predictor d_gap_mean --boot_reps 5000 --seed 1337
python .\pass44_seedlevel_pooled_regression.py --pairs_with_seed_spectral_csv $O44\pass44_pairs_with_seed_spectral.csv --outdir $O44 --predictor d_gap_std  --boot_reps 5000 --seed 1337
If ΔR² CI excludes 0 for d_gap_cv, you’ve got real evidence that eigenspace spacing irregularity predicts invariance breakdown beyond N-scaling.