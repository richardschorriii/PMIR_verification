#!/usr/bin/env python3
"""
PASS6 - PMIR Verification Test
Extracted from ChatGPT transcript (line 39759)
Length: 8787 characters
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------- utils ----------------

def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.check_call([str(x) for x in cmd], cwd=str(cwd) if cwd else None)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_run_dir(out_root: Path) -> Path:
    hits = sorted(out_root.rglob("curve.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(f"No curve.csv found under {out_root}")
    return hits[0].parent

def stage_curve_for_tauhalf(curve_csv: Path, stage_dir: Path, N: int, label: str) -> Path:
    ensure_dir(stage_dir)
    staged = stage_dir / f"N{N}_{label}.csv"
    shutil.copy2(curve_csv, staged)
    return staged

def run_tauhalf(base_stage: Path, probe_stage: Path, outdir: Path,
               tmin: float, tmax: float, grid_dt: float,
               abs_delta: bool, normalize: bool) -> Path:
    ensure_dir(outdir)
    cmd = [
        "python", ".\\pmir_probe_tauhalf.py",
        "--base_dir", str(base_stage),
        "--probe_dir", str(probe_stage),
        "--tmin", str(tmin),
        "--tmax", str(tmax),
        "--grid_dt", str(grid_dt),
        "--outdir", str(outdir),
    ]
    if abs_delta:
        cmd.append("--abs_delta")
    if normalize:
        cmd.append("--normalize")
    run(cmd)
    out_csv = outdir / "tauhalf_summary.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected {out_csv} not found")
    return out_csv

def parse_tauhalf_summary(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty tauhalf summary: {path}")
    r = rows[0]
    out: Dict[str, float] = {}
    for k, v in r.items():
        if v is None:
            out[k] = float("nan")
            continue
        s = str(v).strip()
        if s == "":
            out[k] = float("nan")
            continue
        try:
            out[k] = float(s)
        except Exception:
            # keep non-numeric as-is (rare)
            out[k] = float("nan")
    return out

def tauhalf_lower_bound(tau_row: Dict[str, float], tmax_used: float) -> float:
    cens = tau_row.get("censored", float("nan"))
    t_peak = tau_row.get("t_peak", float("nan"))
    if not (math.isfinite(tmax_used) and math.isfinite(t_peak)):
        return float("nan")
    if math.isfinite(cens) and int(cens) == 1:
        return float(tmax_used - t_peak)
    th = tau_row.get("tau_half", float("nan"))
    return float(th) if math.isfinite(th) else float("nan")

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# ---------------- PMIR curve generation ----------------

def gen_curve(out_root: Path,
              topology: str,
              N: int,
              degree: int,
              graph_seed: int,
              pmir_seed: int,
              T: float, dt: float,
              gamma: float, beta: float,
              obs: str,
              probe_eps: float,
              probe_kind: str,
              probe_mode: str,
              probe_dir_mode: str) -> Tuple[Path, Path]:
    ensure_dir(out_root)
    cmd = [
        "python", ".\\pmir_generate_rivalry_curves.py",
        "--topology", topology,
        "--N", str(N),
        "--degree", str(degree),
        "--graph_seed", str(graph_seed),
        "--pmir_seed", str(pmir_seed),
        "--n_graphs", "1",
        "--n_seeds", "1",
        "--T", str(T),
        "--dt", str(dt),
        "--gamma", str(gamma),
        "--beta", str(beta),
        "--obs", obs,
        "--probe_eps", str(probe_eps),
        "--probe_kind", probe_kind,
        "--probe_mode", probe_mode,
        "--probe_dir_mode", probe_dir_mode,
        "--out_root", str(out_root),
    ]
    run(cmd)
    run_dir = find_latest_run_dir(out_root)
    curve_csv = run_dir / "curve.csv"
    meta_json = run_dir / "meta.json"
    if not curve_csv.exists():
        raise FileNotFoundError(f"Missing {curve_csv}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing {meta_json}")
    return curve_csv, meta_json

# ---------------- analysis helpers ----------------

def effect_symmetry(a: float, b: float) -> float:
    """
    Symmetry score: |a - b| / (|a| + |b| + eps)
    0 = perfectly symmetric, 1 ~ very asymmetric.
    """
    eps = 1e-12
    return abs(a - b) / (abs(a) + abs(b) + eps)

def write_csv(path: Path, header: List[str], rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)

    # sweep choices
    ap.add_argument("--topology", type=str, default="rr")  # run rr first, then grid
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--graph_seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--pmir_seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--eps", type=float, default=0.05)

    # PMIR sim params
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--obs", type=str, default="l1")

    # probe params (match your canon)
    ap.add_argument("--probe_kind", type=str, default="step")
    ap.add_argument("--probe_mode", type=str, default="add_to_dx")
    ap.add_argument("--probe_dir_mode", type=str, default="fiedler")

    # tauhalf window
    ap.add_argument("--tmin", type=float, default=2.0)
    ap.add_argument("--tmax", type=float, default=96.0)
    ap.add_argument("--grid_dt", type=float, default=0.05)
    ap.add_argument("--abs_delta", action="store_true", default=False)
    ap.add_argument("--normalize", action="store_true", default=False)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    pass_root = out_root / "pass7_causality"
    runs_root = out_root / "runs_pass7"
    stage_root = pass_root / "_stage_tauhalf"
    tau_root = pass_root / "tauhalf"
    ensure_dir(pass_root); ensure_dir(runs_root); ensure_dir(stage_root); ensure_dir(tau_root)

    graph_seeds = [int(x) for x in args.graph_seeds.split(",") if x.strip()]
    pmir_seeds = [int(x) for x in args.pmir_seeds.split(",") if x.strip()]

    rows = []
    for g in graph_seeds:
        for p in pmir_seeds:
            # --- baseline (eps=0) ---
            base_out = runs_root / f"BASE_topo={args.topology}_N{args.N}_g{g}_p{p}_eps0"
            base_curve, base_meta = gen_curve(
                out_root=base_out,
                topology=args.topology,
                N=args.N,
                degree=args.degree,
                graph_seed=g,
                pmir_seed=p,
                T=args.T, dt=args.dt,
                gamma=args.gamma, beta=args.beta,
                obs=args.obs,
                probe_eps=0.0,
                probe_kind=args.probe_kind,
                probe_mode=args.probe_mode,
                probe_dir_mode=args.probe_dir_mode,
            )
            lam2 = safe_float(read_json(base_meta).get("lambda2", float("nan")))

            # --- +eps probe ---
            plus_out = runs_root / f"PROBEPLUS_topo={args.topology}_N{args.N}_g{g}_p{p}_eps{args.eps:g}"
            plus_curve, _ = gen_curve(
                out_root=plus_out,
                topology=args.topology,
                N=args.N,
                degree=args.degree,
                graph_seed=g,
                pmir_seed=p,
                T=args.T, dt=args.dt,
                gamma=args.gamma, beta=args.beta,
                obs=args.obs,
                probe_eps=+abs(args.eps),
                probe_kind=args.probe_kind,
                probe_mode=args.probe_mode,
                probe_dir_mode=args.probe_dir_mode,
            )

            # --- -eps probe ---
            minus_out = runs_root / f"PROBEMINUS_topo={args.topology}_N{args.N}_g{g}_p{p}_eps{-abs(args.eps):g}"
            minus_curve, _ = gen_curve(
                out_root=minus_out,
                topology=args.topology,
                N=args.N,
                degree=args.degree,
                graph_seed=g,