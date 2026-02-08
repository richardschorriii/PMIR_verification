#!/usr/bin/env python3
"""
PASS0 - PMIR Verification Test
Extracted from ChatGPT transcript (line 34881)
Length: 10722 characters
"""

import pandas as pd
        import numpy as np

        if master_csv.exists():
            df = pd.read_csv(master_csv)
        else:
            df = pd.DataFrame()

        # focus: baseline pass only, eps=0.05 by default if present
        eps_target = 0.05
        d = df[(df["pass_id"] == "P0_BASELINE")].copy()
        for c in ["probe_eps","lambda2_base","tau_half","N"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["tau_half","lambda2_base","probe_eps","N"])
        d = d[(d["lambda2_base"] > 0) & np.isfinite(d["lambda2_base"]) & np.isfinite(d["tau_half"])]

        d = d[np.isclose(d["probe_eps"].to_numpy(), eps_target)]

        lines = []
        lines.append(f"PASS1: lambda2 explainability (no cross-topology matching)")
        lines.append(f"Using eps_target={eps_target}")
        lines.append(f"Rows used: {len(d)}")
        lines.append("")

        if len(d) < 20:
            lines.append("Not enough rows; run more P0_BASELINE first.")
        else:
            d["x"] = np.log(1.0 / d["lambda2_base"].astype(float))
            # within-topology fits
            for topo, s in d.groupby("topology"):
                if len(s) < 10:
                    continue
                x = s["x"].to_numpy()
                y = s["tau_half"].to_numpy()
                A = np.c_[np.ones_like(x), x]
                c = np.linalg.lstsq(A, y, rcond=None)[0]
                yhat = A @ c
                r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                lines.append(f"{topo}: tau ≈ {c[0]:.6f} + {c[1]:.6f}·log(1/lambda2)   R²={r2:.4f}   n={len(s)}")

            # pooled model with topology dummy (quick)
            # tau ~ a + b*log(1/l2) + c*I[topo==grid]
            if set(d["topology"].unique()) >= {"grid2d_periodic","rr"}:
                x = d["x"].to_numpy()
                y = d["tau_half"].to_numpy()
                is_grid = (d["topology"] == "grid2d_periodic").astype(float).to_numpy()
                A = np.c_[np.ones_like(x), x, is_grid]
                c = np.linalg.lstsq(A, y, rcond=None)[0]
                yhat = A @ c
                r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                lines.append("")
                lines.append(f"Pooled: tau ≈ {c[0]:.6f} + {c[1]:.6f}·log(1/lambda2) + {c[2]:.6f}·I[grid]   R²={r2:.4f}")
                lines.append(f"Interpretation: c2 is the topology effect after accounting for lambda2.")

        pass1_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Wrote: {pass1_path}")
    except Exception as e:
        print(f"[WARN] PASS1 explainability step failed: {e}")
    # ---- PASS 2: probe invariance ----
    if args.do_probe_invariance:
        pass_id = "P2_PROBE_INVARIANCE"
        dir_modes = [x.strip() for x in args.probe_dir_modes.split(",") if x.strip()]
        probe_modes = [x.strip() for x in args.probe_modes.split(",") if x.strip()]

        for topo in ["grid2d_periodic", "rr"]:
            for N in Ns:
                for g in graph_seeds:
                    for p in pmir_seeds:
                        for dm in dir_modes:
                            for pm in probe_modes:
                                base_out = out_root / "runs" / f"{pass_id}_topo={topo}_N{N}_g{g}_p{p}_dm={dm}_pm={pm}_eps0"
                                base_curve, base_meta = gen_curve(
                                    out_root=base_out,
                                    topology=topo, N=N, degree=args.degree,
                                    graph_seed=g, pmir_seed=p,
                                    T=args.T, dt=args.dt,
                                    gamma=args.gamma, beta=args.beta,
                                    obs=args.obs,
                                    probe_eps=0.0,
                                    probe_kind=args.probe_kind,
                                    probe_mode=pm,
                                    probe_dir_mode=dm,
                                    save_states=False,
                                )
                                meta = read_meta(base_meta)
                                lam2 = float(meta.get("lambda2", float("nan")))

                                for eps in eps_list:
                                    probe_out = out_root / "runs" / f"{pass_id}_topo={topo}_N{N}_g{g}_p{p}_dm={dm}_pm={pm}_eps{eps}"
                                    probe_curve, _ = gen_curve(
                                        out_root=probe_out,
                                        topology=topo, N=N, degree=args.degree,
                                        graph_seed=g, pmir_seed=p,
                                        T=args.T, dt=args.dt,
                                        gamma=args.gamma, beta=args.beta,
                                        obs=args.obs,
                                        probe_eps=eps,
                                        probe_kind=args.probe_kind,
                                        probe_mode=pm,
                                        probe_dir_mode=dm,
                                        save_states=False,
                                    )

                                    base_stage = stage_root / f"{pass_id}_base_{topo}_N{N}_g{g}_p{p}_dm={dm}_pm={pm}"
                                    probe_stage = stage_root / f"{pass_id}_probe_{topo}_N{N}_g{g}_p{p}_dm={dm}_pm={pm}_eps{eps}"
                                    stage_curve_for_tauhalf(base_curve, base_stage, N, "curve")
                                    stage_curve_for_tauhalf(probe_curve, probe_stage, N, "curve")

                                    tau_outdir = out_root / "tauhalf" / pass_id / f"topo={topo}" / f"N={N}" / f"g={g}" / f"p={p}" / f"dm={dm}" / f"pm={pm}" / f"eps={eps:.5f}"
                                    tau_csv = run_tauhalf(base_stage, probe_stage, tau_outdir, args.tmin, args.tmax, args.grid_dt, args.abs_delta, args.normalize)
                                    tau = parse_tauhalf_summary(tau_csv)
                                    cens = int(tau.get("censored", 0)) if math.isfinite(tau.get("censored", 0)) else ""
                                    tau_lb = tauhalf_lower_bound(tau, args.tmax)

                                    row = dict(
                                        ...
                                        tmin_used=args.tmin,
                                        tmax_used=args.tmax,
                                        censored=cens,
                                        tau_half=tau.get("tau_half", ""),
                                        tau_half_lb=tau_lb,
                                        peak_delta=tau.get("peak_delta", ""),
                                        t_peak=tau.get("t_peak", ""),
                                        auc_delta=tau.get("auc_delta", ""),
                                        ...
                                    )
                                    write_row(master_csv, header, dict(
                                        pass_id=pass_id, topology=topo, N=N, graph_seed=g, pmir_seed=p,
                                        probe_eps=eps, probe_kind=args.probe_kind, probe_mode=pm, probe_dir_mode=dm,
                                        lambda2_base=lam2,
                                        tau_half=tau.get("tau_half",""), peak_delta=tau.get("peak_delta",""),
                                        t_peak=tau.get("t_peak",""), auc_delta=tau.get("auc_delta",""),
                                        base_curve=str(base_curve), probe_curve=str(probe_curve),
                                        notes=""
                                    ))

    # ---- Stats summary (minimal, robust) ----
    # Produces: tauhalf_master_stats.txt
    stats_path = out_root / "tauhalf_master_stats.txt"
    try:
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