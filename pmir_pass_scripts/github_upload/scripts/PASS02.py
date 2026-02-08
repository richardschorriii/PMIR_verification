#!/usr/bin/env python3
"""
PASS2 - PMIR Verification Test
Extracted from ChatGPT transcript (line 30779)
Length: 3377 characters
"""

import pandas as pd
        df = pd.read_csv(master_csv)
        for c in ["N","graph_seed","pmir_seed","probe_eps","lambda2_base","tau_half","peak_delta","t_peak","auc_delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # primary apples-to-apples: eps=0.05 if present
        eps_target = 0.05
        d = df[(df["pass_id"] == "P0_BASELINE") & (df["probe_eps"].sub(eps_target).abs() < 1e-12)].dropna(subset=["tau_half"])
        lines = []
        lines.append(f"MASTER CSV: {master_csv}")
        lines.append(f"Rows total: {len(df)}")
        lines.append("")
        lines.append("=== P0_BASELINE summary @ eps=0.05 ===")
        if len(d) == 0:
            lines.append("No rows found for P0_BASELINE eps=0.05")
        else:
            for topo, sub in d.groupby("topology"):
                lines.append(f"{topo}: n={len(sub)} mean_tau={sub['tau_half'].mean():.6f} std_tau={sub['tau_half'].std(ddof=1):.6f}")
            if set(d["topology"].unique()) >= {"grid2d_periodic","rr"}:
                A = d[d.topology=="grid2d_periodic"]["tau_half"].tolist()
                B = d[d.topology=="rr"]["tau_half"].tolist()
                lines.append("")
                lines.append(f"delta_mean(grid-rr) = {(sum(A)/len(A) - sum(B)/len(B)):.6f}")
                lines.append(f"cohen_d = {cohen_d(A,B):.6f}")
                lines.append(f"mannwhitney_p = {mann_whitney_p(A,B):.6e}")

        # “is it just lambda2?” quick sanity
        lines.append("")
        lines.append("=== lambda2 explanatory power (quick) ===")
        # correlate tau with log(1/lambda2) for P0 baseline
        sub = df[df["pass_id"]=="P0_BASELINE"].dropna(subset=["tau_half","lambda2_base"])
        sub = sub[(sub["lambda2_base"] > 0)]
        if len(sub) >= 20:
            import numpy as np
            sub = sub.copy()
            sub["x"] = np.log(1.0 / sub["lambda2_base"].astype(float))
            for topo, s in sub.groupby("topology"):
                if len(s) < 20:
                    continue
                x = s["x"].to_numpy()
                y = s["tau_half"].to_numpy()
                A_ = np.c_[np.ones_like(x), x]
                c = np.linalg.lstsq(A_, y, rcond=None)[0]
                yhat = A_ @ c
                r2 = 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()
                lines.append(f"{topo}: tau ≈ {c[0]:.6f} + {c[1]:.6f}·log(1/lambda2)  R²={r2:.4f}  n={len(s)}")
        else:
            lines.append("Not enough rows for correlation.")

        stats_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Wrote: {stats_path}")
    except Exception as e:
        print(f"[WARN] Stats step failed: {e}")
        print(f"[WARN] master still written: {master_csv}")

if __name__ == "__main__":
    main()
2) PowerShell commands (copy/paste)
A) Clean baseline TIME_LOCK (fast + canonical)
cd C:\Users\veilbreaker\String_Theory

python .\pmir_master_time_lock.py `
  --out_root .\analysis_transfer\PMIR_TIME_LOCK_MASTER `
  --Ns "1024,2048,4096" `
  --degree 4 `
  --graph_seeds "0,1,2,3,4" `
  --pmir_seeds "0,1,2,3,4" `
  --eps_list "0.05" `
  --topologies "grid2d_periodic,rr" `
  --tmin 2 --tmax 48 --grid_dt 0.05
Outputs you care about:

.\analysis_transfer\PMIR_TIME_LOCK_MASTER\tauhalf_master.csv

.\analysis_transfer\PMIR_TIME_LOCK_MASTER\tauhalf_master_stats.txt