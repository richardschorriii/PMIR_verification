#!/usr/bin/env python3
"""
PASS40 - PMIR Verification Test
Extracted from ChatGPT transcript (line 88869)
Length: 9813 characters
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ols_xy(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    den = np.sum((y - y.mean())**2)
    r2 = 1.0 - (np.sum((y - yhat)**2) / den) if den > 0 else np.nan
    return b, r2


def cv(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(m) or m == 0:
        return np.nan
    return s / abs(m)


def alpha_grid_from_str(s: str):
    # accepts "0:1:0.05" or "0,0.1,0.2"
    s = s.strip()
    if ":" in s:
        a, b, step = [float(x) for x in s.split(":")]
        if step <= 0:
            raise ValueError("alpha step must be >0")
        # include endpoint
        n = int(np.floor((b - a) / step + 1e-9)) + 1
        grid = a + step * np.arange(n)
        # clamp last
        if grid[-1] < b - 1e-9:
            grid = np.append(grid, b)
        return np.clip(grid, 0.0, 1.0)
    else:
        return np.array([float(x) for x in s.split(",") if x.strip() != ""], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="PASS33 summary_by_dir_probe_topoN_eps.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topo_ref", default="rr")
    ap.add_argument("--topo_alt", default="grid2d_periodic")

    ap.add_argument("--dir_base", default="fiedler")
    ap.add_argument("--dir_competitors", default="smooth,random")

    ap.add_argument("--use_abs", type=int, default=1)

    ap.add_argument("--alpha_grid", default="0:1:0.05")
    ap.add_argument("--N_ref", type=float, default=2048.0)

    ap.add_argument("--beta_tol", type=float, default=0.10)
    ap.add_argument("--cv_tol", type=float, default=0.50)
    ap.add_argument("--growth_tol", type=float, default=2.0)

    ap.add_argument("--boot_reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.outdir)
    ensure_dir(out)

    df = pd.read_csv(args.in_csv)

    need = ["probe_dir_mode", "probe_mode", "topology", "N", "probe_eps", "mean_score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing columns: {miss}")

    # clean types
    df = df.dropna(subset=need).copy()
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["probe_eps"] = pd.to_numeric(df["probe_eps"], errors="coerce")
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df = df.dropna(subset=["N", "probe_eps", "mean_score"])

    if args.use_abs:
        df["score"] = df["mean_score"].abs()
    else:
        df["score"] = df["mean_score"]

    # We will operate per (probe_mode, probe_eps, N) and direction
    # Make a wide table over dir_mode for each topology
    key = ["probe_mode", "probe_eps", "topology", "N"]

    # sanity: ensure we have base direction present
    dirs_present = sorted(df["probe_dir_mode"].unique().tolist())
    if args.dir_base not in dirs_present:
        raise SystemExit(f"[ERR] dir_base={args.dir_base} not in available dirs: {dirs_present}")

    competitors = [d.strip() for d in args.dir_competitors.split(",") if d.strip()]
    for d in competitors:
        if d not in dirs_present:
            raise SystemExit(f"[ERR] competitor dir={d} not in available dirs: {dirs_present}")

    alphas = alpha_grid_from_str(args.alpha_grid)
    rng = np.random.default_rng(args.seed)

    rows_sweep = []
    rows_break = []

    # loop competitors and all (probe_mode, probe_eps)
    for dcomp in competitors:
        # build base/comp scores for each topology separately
        sub_base = df[df["probe_dir_mode"] == args.dir_base][key + ["score"]].rename(columns={"score": "score_base"})
        sub_comp = df[df["probe_dir_mode"] == dcomp][key + ["score"]].rename(columns={"score": "score_comp"})

        M = sub_base.merge(sub_comp, on=key, how="inner")
        if len(M) == 0:
            print(f"[WARN] no merged rows for base={args.dir_base} comp={dcomp}")
            continue

        # split topo ref/alt and pair them
        K = ["probe_mode", "probe_eps", "N"]
        A = M[M["topology"] == args.topo_ref][K + ["score_base", "score_comp"]].rename(
            columns={"score_base": "ref_base", "score_comp": "ref_comp"}
        )
        B = M[M["topology"] == args.topo_alt][K + ["score_base", "score_comp"]].rename(
            columns={"score_base": "alt_base", "score_comp": "alt_comp"}
        )
        P = A.merge(B, on=K, how="inner")
        if len(P) == 0:
            print(f"[WARN] no paired rows for base={args.dir_base} comp={dcomp}")
            continue

        # per (probe_mode, probe_eps) run alpha sweep
        for (pm, eps), G in P.groupby(["probe_mode", "probe_eps"]):
            # Need at least 3 N points (you have N=1024,2048,4096)
            Ns = np.array(sorted(G["N"].unique()))
            if len(Ns) < 3:
                continue

            # collect sweep results
            sweep_tmp = []

            # for bootstrapping we fit logR vs logN on the 3 points using the *per-N* ratio means
            for alpha in alphas:
                # mixed scores for each row
                ref = (1 - alpha) * G["ref_base"].to_numpy(dtype=float) + alpha * G["ref_comp"].to_numpy(dtype=float)
                alt = (1 - alpha) * G["alt_base"].to_numpy(dtype=float) + alpha * G["alt_comp"].to_numpy(dtype=float)

                # protect
                ref = np.where(ref == 0, np.nan, ref)
                ratio = alt / ref

                tmp = pd.DataFrame({"N": G["N"].to_numpy(dtype=float), "ratio": ratio})
                gN = tmp.groupby("N", as_index=False).agg(R=("ratio", "mean")).sort_values("N")

                # log fit
                x = np.log(gN["N"].to_numpy(dtype=float))
                y = np.log(gN["R"].to_numpy(dtype=float))
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3:
                    continue

                b, r2 = ols_xy(x[ok], y[ok])
                beta = float(b[1])
                intercept = float(b[0])

                # bootstrap beta
                xok = x[ok].astype(float)
                yok = y[ok].astype(float)
                betas = np.empty(args.boot_reps, dtype=float)
                npts = len(xok)
                for i in range(args.boot_reps):
                    idx = rng.integers(0, npts, npts)
                    bb, _ = ols_xy(xok[idx], yok[idx])
                    betas[i] = bb[1]
                beta_lo, beta_hi = np.quantile(betas, [0.025, 0.975])

                # diagnostics
                R_min = float(np.nanmin(gN["R"].to_numpy(dtype=float)))
                R_max = float(np.nanmax(gN["R"].to_numpy(dtype=float)))
                R_growth = (R_max / R_min) if (np.isfinite(R_min) and R_min != 0) else np.nan
                ratio_cv = cv(gN["R"].to_numpy(dtype=float))
                R_at = float(np.exp(intercept + beta * np.log(args.N_ref)))

                sweep_tmp.append(dict(
                    dir_base=args.dir_base,
                    dir_comp=dcomp,
                    probe_mode=pm,
                    probe_eps=float(eps),
                    alpha=float(alpha),
                    N_unique=int(ok.sum()),
                    beta_logR_logN=beta,
                    beta_lo=float(beta_lo),
                    beta_hi=float(beta_hi),
                    ci_crosses_0=bool((beta_lo <= 0) and (beta_hi >= 0)),
                    r2_logfit=float(r2),
                    ratio_cv=float(ratio_cv),
                    R_growth=float(R_growth),
                    R_at_Nref=float(R_at),
                ))

            if not sweep_tmp:
                continue

            S = pd.DataFrame(sweep_tmp).sort_values("alpha").reset_index(drop=True)
            for _, r in S.iterrows():
                rows_sweep.append(r.to_dict())

            # break rule: first alpha where (CI excludes 0) OR ratio_cv > cv_tol OR R_growth > growth_tol
            break_alpha = None
            for _, r in S.iterrows():
                ci_excludes_0 = not bool(r["ci_crosses_0"])
                if ci_excludes_0 or (r["ratio_cv"] > args.cv_tol) or (r["R_growth"] > args.growth_tol):
                    break_alpha = float(r["alpha"])
                    break

            # also store baseline alpha=0 metrics
            r0 = S.iloc[0].to_dict()
            rows_break.append(dict(
                dir_base=args.dir_base,
                dir_comp=dcomp,
                probe_mode=pm,
                probe_eps=float(eps),
                alpha_break=break_alpha,
                beta0=float(r0["beta_logR_logN"]),
                beta0_lo=float(r0["beta_lo"]),
                beta0_hi=float(r0["beta_hi"]),
                ratio_cv0=float(r0["ratio_cv"]),
                R_growth0=float(r0["R_growth"]),
                r2_0=float(r0["r2_logfit"]),
                notes="break if CI excludes 0 OR ratio_cv>cv_tol OR R_growth>growth_tol"
            ))

    SW = pd.DataFrame(rows_sweep)
    BR = pd.DataFrame(rows_break)

    out_sw = out / "pass41_alpha_sweep.csv"
    out_br = out / "pass41_alpha_breaks.csv"
    out_txt = out / "pass41_summary.txt"

    if len(SW) == 0:
        raise SystemExit("[ERR] produced no sweep rows (check dirs/topos)")

    SW.to_csv(out_sw, index=False)
    BR.to_csv(out_br, index=False)

    # compact summary
    # prioritize small alpha_break (fragile) vs None (robust up to 1)
    BR2 = BR.copy()
    BR2["alpha_break_rank"] = BR2["alpha_break"].fillna(2.0)
    BR2 = BR2.sort_values(["alpha_break_rank", "dir_comp", "probe_mode", "probe_eps"]).reset_index(drop=True)