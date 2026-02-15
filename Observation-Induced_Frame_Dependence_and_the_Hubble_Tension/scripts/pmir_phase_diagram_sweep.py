"""
PMIR Observation Map Phase Diagram Sweep
=========================================
Author: Richard L. Schorr III
Purpose: Systematic sweep of parameterized observation maps to locate the
         phase boundary between hierarchy-inducing and non-hierarchy-inducing
         observation frames.

Structural Gate (from Preprints 1 & Hierarchical Persistence paper):
  G: (3.0 ≤ β ≤ 5.5) ∧ (entropy_slope < 0) ∧ (R² > 0.35) ∧ (curvature_persistence > 0.30)

Map families swept:
  1. Scaled exponential:   g(z; α) = exp(α * z)              α ∈ [0.1, 3.0]
  2. Softplus:             g(z; k) = (1/k) * log(1 + exp(k*z))  k ∈ [0.1, 10.0]
  3. Power law (abs):      g(z; p) = |z|^p                    p ∈ [0.5, 4.0]
  4. Sigmoid-scaled:       g(z; s) = exp(s * sigmoid(z))      s ∈ [0.1, 6.0]
  5. Box-Cox family:       g(z; λ) = (exp(λz) - 1)/λ  → exp as λ→0   λ ∈ [0.01, 2.0]

For each map + parameters, sweep latent process (λ_mem, σ) space.

Outputs:
  - pmir_phase_diagram_results.csv  (full results table)
  - pmir_phase_diagram_expfamily.png  (α sweep — exponential family)
  - pmir_phase_diagram_softplus.png
  - pmir_phase_diagram_powerlaw.png
  - pmir_phase_diagram_composite.png  (all maps, boundary overlay)
  - pmir_hubble_frame_comparison.png  (CMB-like vs distance-ladder-like frames)
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from itertools import product

# ─────────────────────────────────────────────
# SECTION 1: LATENT PROCESS GENERATOR
# ─────────────────────────────────────────────

def generate_latent(lam_mem, sigma, T=5000, seed=42):
    """
    AR(1) latent process: z_{t+1} = lam_mem * z_t + sigma * xi_t
    lam_mem: memory parameter (0 = no memory, →1 = long memory)
    sigma:   noise scale
    T:       time series length
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = lam_mem * z[t-1] + sigma * rng.standard_normal()
    return z


# ─────────────────────────────────────────────
# SECTION 2: OBSERVATION MAP FAMILIES
# ─────────────────────────────────────────────

def map_scaled_exp(z, alpha):
    """g(z; α) = exp(α*z). α=1 is the canonical exponential map."""
    return np.exp(np.clip(alpha * z, -50, 50))

def map_softplus(z, k):
    """g(z; k) = (1/k)*log(1 + exp(k*z)). Approaches linear as k→0, exp as k→∞."""
    kz = np.clip(k * z, -500, 500)
    return (1.0 / k) * np.log1p(np.exp(kz))

def map_powerlaw(z, p):
    """g(z; p) = |z|^p. Never exponential; tests non-exp family."""
    return np.power(np.abs(z) + 1e-10, p)

def map_sigmoid_scaled(z, s):
    """g(z; s) = exp(s * sigmoid(z)). Bounded exponential — tests clipping effect."""
    sig = 1.0 / (1.0 + np.exp(-z))
    return np.exp(s * sig)

def map_boxcox(z, lam):
    """
    Box-Cox family: g(z; λ) = (exp(λz) - 1)/λ
    As λ→0, this approaches z (linear).
    At λ=1, this is (exp(z) - 1).
    Bridges linear and exponential continuously.
    """
    if lam < 1e-4:
        return z.copy()  # linear limit
    return (np.exp(np.clip(lam * z, -50, 50)) - 1.0) / lam

# Registry of all map families with their parameter ranges and labels
MAP_FAMILIES = {
    'scaled_exp': {
        'func': map_scaled_exp,
        'param_name': 'alpha',
        'param_range': np.linspace(0.1, 3.0, 30),
        'label': r'Scaled Exponential: $g(z;\alpha)=e^{\alpha z}$',
        'xlabel': r'Exponent scale $\alpha$',
    },
    'softplus': {
        'func': map_softplus,
        'param_name': 'k',
        'param_range': np.logspace(-1, 1, 30),  # 0.1 to 10
        'label': r'Softplus: $g(z;k)=\frac{1}{k}\ln(1+e^{kz})$',
        'xlabel': r'Sharpness parameter $k$',
    },
    'powerlaw': {
        'func': map_powerlaw,
        'param_name': 'p',
        'param_range': np.linspace(0.5, 4.0, 30),
        'label': r'Power Law: $g(z;p)=|z|^p$',
        'xlabel': r'Power exponent $p$',
    },
    'sigmoid_scaled': {
        'func': map_sigmoid_scaled,
        'param_name': 's',
        'param_range': np.linspace(0.1, 8.0, 30),
        'label': r'Sigmoid-Scaled: $g(z;s)=e^{s\cdot\sigma(z)}$',
        'xlabel': r'Scale $s$',
    },
    'boxcox': {
        'func': map_boxcox,
        'param_name': 'lambda',
        'param_range': np.concatenate([np.linspace(0.01, 0.1, 5),
                                        np.linspace(0.1, 2.0, 25)]),
        'label': r'Box-Cox: $g(z;\lambda)=\frac{e^{\lambda z}-1}{\lambda}$',
        'xlabel': r'Box-Cox parameter $\lambda$',
    },
}


# ─────────────────────────────────────────────
# SECTION 3: GATE METRIC COMPUTATION
# ─────────────────────────────────────────────

def compute_tail_exponent(series, tail_fraction=0.1):
    """
    Estimate tail exponent β from upper-tail log-log CCDF fit.
    Returns β and quality of fit R².
    Lower β = heavier tail. Gate: 3.0 ≤ β ≤ 5.5
    """
    s = series[series > 0]
    if len(s) < 50:
        return np.nan, 0.0

    threshold = np.quantile(s, 1.0 - tail_fraction)
    tail = s[s >= threshold]

    if len(tail) < 10:
        return np.nan, 0.0

    tail_sorted = np.sort(tail)[::-1]
    n = len(tail_sorted)
    rank = np.arange(1, n + 1)
    ccdf = rank / n

    log_x = np.log(tail_sorted)
    log_ccdf = np.log(ccdf)

    # Only keep finite values
    mask = np.isfinite(log_x) & np.isfinite(log_ccdf)
    if mask.sum() < 5:
        return np.nan, 0.0

    slope, intercept, r, p, se = stats.linregress(log_x[mask], log_ccdf[mask])
    beta = -slope  # CCDF slope is negative of tail exponent
    r2 = r**2
    return beta, r2


def compute_entropy_gradient(series, n_bins=20):
    """
    Scale-conditional entropy: Shannon entropy within log-spaced bins.
    Fit entropy vs log(scale). Returns slope and R².
    Gate: slope < 0, R² > 0.35
    """
    s = series[series > 0]
    if len(s) < 100:
        return np.nan, 0.0

    log_s = np.log(s)
    log_min, log_max = np.percentile(log_s, 1), np.percentile(log_s, 99)
    if log_max <= log_min:
        return np.nan, 0.0

    # Log-spaced bin edges
    edges = np.linspace(log_min, log_max, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_scales = np.exp(bin_centers)

    entropies = []
    valid_scales = []

    for i in range(n_bins):
        mask = (log_s >= edges[i]) & (log_s < edges[i+1])
        bin_vals = s[mask]
        if len(bin_vals) < 5:
            continue
        # Normalize to get probability distribution within bin
        bin_vals_norm = bin_vals / bin_vals.sum()
        # Shannon entropy
        ent = -np.sum(bin_vals_norm * np.log(bin_vals_norm + 1e-12))
        entropies.append(ent)
        valid_scales.append(bin_scales[i])

    if len(entropies) < 5:
        return np.nan, 0.0

    log_scales = np.log(np.array(valid_scales))
    entropies = np.array(entropies)

    slope, intercept, r, p, se = stats.linregress(log_scales, entropies)
    r2 = r**2
    return slope, r2


def compute_curvature_persistence(series, n_windows=20, window_frac=0.3):
    """
    Sliding-window quadratic fits to log-log CCDF.
    Persistence = fraction of windows with non-negligible curvature (|quadratic coeff| > threshold).
    Gate: persistence > 0.30
    """
    s = series[series > 0]
    if len(s) < 100:
        return 0.0

    s_sorted = np.sort(s)[::-1]
    n = len(s_sorted)
    ccdf = np.arange(1, n + 1) / n

    log_x = np.log(s_sorted)
    log_ccdf = np.log(ccdf)

    mask = np.isfinite(log_x) & np.isfinite(log_ccdf)
    log_x = log_x[mask]
    log_ccdf = log_ccdf[mask]

    if len(log_x) < 20:
        return 0.0

    window_size = max(10, int(len(log_x) * window_frac))
    step = max(1, window_size // 4)
    n_steps = max(1, (len(log_x) - window_size) // step)

    curvature_count = 0
    total_windows = 0

    for i in range(0, len(log_x) - window_size, step):
        wx = log_x[i:i + window_size]
        wy = log_ccdf[i:i + window_size]
        if len(wx) < 5:
            continue
        # Fit quadratic
        try:
            coeffs = np.polyfit(wx, wy, 2)
            # Significant curvature if quadratic coeff is non-negligible
            if abs(coeffs[0]) > 0.05:
                curvature_count += 1
            total_windows += 1
        except:
            continue

    if total_windows == 0:
        return 0.0

    return curvature_count / total_windows


def evaluate_gate(series):
    """
    Full structural gate evaluation.
    Returns dict with all metrics and pass/fail for each criterion and overall.
    """
    beta, beta_r2 = compute_tail_exponent(series)
    entropy_slope, entropy_r2 = compute_entropy_gradient(series)
    curvature_pers = compute_curvature_persistence(series)

    # Individual gate criteria
    gate_beta = (3.0 <= beta <= 5.5) if not np.isnan(beta) else False
    gate_entropy_slope = (entropy_slope < 0) if not np.isnan(entropy_slope) else False
    gate_entropy_r2 = (entropy_r2 > 0.35) if not np.isnan(entropy_r2) else False
    gate_curvature = (curvature_pers > 0.30)

    gate_pass = gate_beta and gate_entropy_slope and gate_entropy_r2 and gate_curvature

    # Continuous score: fraction of criteria met (0–4 → 0.0–1.0)
    score = sum([gate_beta, gate_entropy_slope, gate_entropy_r2, gate_curvature]) / 4.0

    return {
        'beta': beta,
        'beta_r2': beta_r2,
        'entropy_slope': entropy_slope,
        'entropy_r2': entropy_r2,
        'curvature_persistence': curvature_pers,
        'gate_beta': gate_beta,
        'gate_entropy_slope': gate_entropy_slope,
        'gate_entropy_r2': gate_entropy_r2,
        'gate_curvature': gate_curvature,
        'gate_pass': gate_pass,
        'score': score,
    }


# ─────────────────────────────────────────────
# SECTION 4: SINGLE-POINT EVALUATOR
# ─────────────────────────────────────────────

def evaluate_point(map_func, map_param, lam_mem, sigma, T=5000, n_seeds=5):
    """
    Evaluate gate for a single (map, lam_mem, sigma) point.
    Averages over n_seeds for robustness.
    Returns mean metrics across seeds.
    """
    all_metrics = []

    for seed in range(n_seeds):
        z = generate_latent(lam_mem, sigma, T=T, seed=seed)
        obs = map_func(z, map_param)
        # Ensure positive, non-degenerate series
        obs = np.abs(obs) + 1e-10
        # Filter out extreme outliers that could break fitting
        obs = obs[obs < np.percentile(obs, 99.9)]
        if len(obs) < 100:
            continue
        metrics = evaluate_gate(obs)
        all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Average numerical metrics across seeds
    result = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        if not vals:
            result[key] = np.nan
        elif isinstance(vals[0], bool):
            result[key] = sum(vals) / len(vals)  # fraction of seeds passing
        else:
            result[key] = np.nanmean(vals)

    result['gate_pass'] = result['score'] >= 0.75  # pass if ≥75% criteria met on average

    return result


# ─────────────────────────────────────────────
# SECTION 5: MAIN SWEEP FUNCTIONS
# ─────────────────────────────────────────────

# Latent parameter grid
LAM_MEM_VALUES = np.linspace(0.0, 0.99, 15)   # memory strength
SIGMA_VALUES   = np.array([0.3, 0.5, 0.7, 1.0, 1.5])  # noise scales


def sweep_single_family(family_name, T=4000, n_seeds=3, verbose=True):
    """
    Sweep a single map family across its parameter range and the (lam_mem, sigma) grid.
    Returns DataFrame of results.
    """
    family = MAP_FAMILIES[family_name]
    func = family['func']
    param_range = family['param_range']
    param_name = family['param_name']

    rows = []
    total = len(param_range) * len(LAM_MEM_VALUES) * len(SIGMA_VALUES)
    done = 0

    for map_param in param_range:
        for lam_mem in LAM_MEM_VALUES:
            for sigma in SIGMA_VALUES:
                metrics = evaluate_point(func, map_param, lam_mem, sigma, T=T, n_seeds=n_seeds)
                done += 1

                if metrics is None:
                    continue

                row = {
                    'family': family_name,
                    param_name: map_param,
                    'lam_mem': lam_mem,
                    'sigma': sigma,
                }
                row.update(metrics)
                rows.append(row)

                if verbose and done % 50 == 0:
                    pct = 100 * done / total
                    gp = metrics.get('gate_pass', False)
                    sc = metrics.get('score', 0)
                    print(f"  [{pct:5.1f}%] {param_name}={map_param:.3f} "
                          f"lam={lam_mem:.2f} sig={sigma:.2f} | "
                          f"score={sc:.2f} gate={'PASS' if gp else 'fail'}")

    return pd.DataFrame(rows)


def run_full_sweep(families=None, T=4000, n_seeds=3):
    """Run sweep for all (or specified) map families. Returns combined DataFrame."""
    if families is None:
        families = list(MAP_FAMILIES.keys())

    all_dfs = []
    for fname in families:
        print(f"\n{'='*60}")
        print(f"Sweeping family: {fname}")
        print(f"{'='*60}")
        df = sweep_single_family(fname, T=T, n_seeds=n_seeds, verbose=True)
        all_dfs.append(df)
        print(f"  → {len(df)} points, {df['gate_pass'].sum()} passed gate "
              f"({100*df['gate_pass'].mean():.1f}%)")

    return pd.concat(all_dfs, ignore_index=True)


# ─────────────────────────────────────────────
# SECTION 6: PHASE DIAGRAM PLOTTING
# ─────────────────────────────────────────────

def plot_phase_diagram_single_family(df, family_name, sigma_fixed=0.7,
                                      save_path=None):
    """
    2D phase diagram: lam_mem (x) × map_param (y), colored by gate score.
    Contour line marks the phase boundary (score = 0.75).
    """
    family = MAP_FAMILIES[family_name]
    param_name = family['param_name']

    sub = df[(df['family'] == family_name) &
             (np.abs(df['sigma'] - sigma_fixed) < 0.05)].copy()

    if sub.empty:
        print(f"No data for {family_name} at sigma={sigma_fixed}")
        return

    lam_vals = np.sort(sub['lam_mem'].unique())
    param_vals = np.sort(sub[param_name].unique())

    score_grid = np.full((len(param_vals), len(lam_vals)), np.nan)
    gate_grid  = np.full((len(param_vals), len(lam_vals)), np.nan)

    for i, pv in enumerate(param_vals):
        for j, lv in enumerate(lam_vals):
            row = sub[(np.abs(sub[param_name] - pv) < 1e-9) &
                      (np.abs(sub['lam_mem'] - lv) < 1e-9)]
            if not row.empty:
                score_grid[i, j] = row['score'].values[0]
                gate_grid[i, j]  = float(row['gate_pass'].values[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{family['label']}\n(σ = {sigma_fixed})", fontsize=12)

    # Left: continuous score heatmap
    ax = axes[0]
    im = ax.pcolormesh(lam_vals, param_vals, score_grid,
                       cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, label='Gate score (fraction of criteria met)')

    # Phase boundary contour
    try:
        ax.contour(lam_vals, param_vals, score_grid,
                   levels=[0.75], colors='black', linewidths=2, linestyles='--')
    except:
        pass

    ax.set_xlabel(r'Memory parameter $\lambda_{mem}$', fontsize=11)
    ax.set_ylabel(family['xlabel'], fontsize=11)
    ax.set_title('Gate Score (continuous)', fontsize=10)

    # Right: binary pass/fail
    ax2 = axes[1]
    cmap_binary = mcolors.ListedColormap(['#d62728', '#2ca02c'])
    ax2.pcolormesh(lam_vals, param_vals, gate_grid,
                   cmap=cmap_binary, vmin=0, vmax=1, shading='auto')

    legend_elems = [Patch(facecolor='#2ca02c', label='PASS (hierarchy present)'),
                    Patch(facecolor='#d62728', label='FAIL (no hierarchy)')]
    ax2.legend(handles=legend_elems, loc='upper left', fontsize=9)

    ax2.set_xlabel(r'Memory parameter $\lambda_{mem}$', fontsize=11)
    ax2.set_ylabel(family['xlabel'], fontsize=11)
    ax2.set_title('Gate Pass/Fail', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_composite_boundary(df, sigma_fixed=0.7, save_path=None):
    """
    Composite plot: phase boundary curves for all map families on same axes.
    X-axis: memory (lam_mem), Y-axis: normalized map parameter (0–1).
    Shows which maps support hierarchy and at what parameter values.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(MAP_FAMILIES)))
    boundary_data = {}

    for (fname, family), color in zip(MAP_FAMILIES.items(), colors):
        param_name = family['param_name']
        sub = df[(df['family'] == fname) &
                 (np.abs(df['sigma'] - sigma_fixed) < 0.05)].copy()

        if sub.empty:
            continue

        lam_vals  = np.sort(sub['lam_mem'].unique())
        param_min = sub[param_name].min()
        param_max = sub[param_name].max()

        # For each lam_mem, find the minimum map_param that passes gate
        boundary_lam  = []
        boundary_norm = []

        for lv in lam_vals:
            row_lam = sub[np.abs(sub['lam_mem'] - lv) < 1e-9].sort_values(param_name)
            passing = row_lam[row_lam['gate_pass'] == True]
            if passing.empty:
                continue
            min_passing_param = passing[param_name].min()
            # Normalize to [0, 1]
            norm_val = (min_passing_param - param_min) / (param_max - param_min + 1e-10)
            boundary_lam.append(lv)
            boundary_norm.append(norm_val)

        if boundary_lam:
            ax.plot(boundary_lam, boundary_norm, 'o-', color=color,
                    label=f"{fname} ({param_name})", lw=2, ms=5)
            boundary_data[fname] = (boundary_lam, boundary_norm)

    ax.axhline(0.0, color='gray', ls=':', alpha=0.5)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.axvline(0.5, color='gray', ls='--', alpha=0.3, label='λ_mem = 0.5')

    ax.set_xlabel(r'Memory parameter $\lambda_{mem}$', fontsize=12)
    ax.set_ylabel('Normalized minimum map parameter to pass gate', fontsize=11)
    ax.set_title('Phase Boundary Across Map Families\n'
                 '(lower = hierarchy achieved at weaker map; '
                 'missing = never passes)',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return boundary_data


def plot_hubble_frame_comparison(df, save_path=None):
    """
    Special plot directly connecting to Hubble tension.
    Compares two specific observation frame archetypes:
      - 'CMB-like' frame: large-scale integrating → scaled_exp with high alpha
      - 'Distance ladder' frame: local hierarchical → softplus with high k
    Shows how the same latent process (lam_mem, sigma) space maps to different
    gate outcomes depending on which frame is applied.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Hubble Tension Frame Comparison\n'
                 'Same latent process, different observation maps → different effective realities',
                 fontsize=12)

    sigma_fixed = 0.7

    # Frame A: CMB-like (high-alpha scaled exponential — collective large-scale modes)
    # Frame B: Distance-ladder-like (softplus, moderate k — local hierarchical)
    # Frame C: Difference map (where they disagree)

    def get_gate_grid(family_name, param_val):
        param_name = MAP_FAMILIES[family_name]['param_name']
        sub = df[(df['family'] == family_name) &
                 (np.abs(df['sigma'] - sigma_fixed) < 0.05)]
        sub = sub[np.abs(sub[param_name] - param_val) < (
            MAP_FAMILIES[family_name]['param_range'][1] -
            MAP_FAMILIES[family_name]['param_range'][0]) * 0.6]

        lam_vals   = np.sort(df['lam_mem'].unique())
        gate_grid  = np.full(len(lam_vals), np.nan)

        for j, lv in enumerate(lam_vals):
            row = sub[np.abs(sub['lam_mem'] - lv) < 1e-9]
            if not row.empty:
                gate_grid[j] = float(row['gate_pass'].values[0])

        return lam_vals, gate_grid

    # Pick representative parameter values
    # CMB-like: high alpha in scaled_exp (strong integrating frame)
    cmb_alpha = MAP_FAMILIES['scaled_exp']['param_range'][-8]  # high alpha
    # Distance-ladder: moderate k in softplus (moderate sharpness)
    dl_k = MAP_FAMILIES['softplus']['param_range'][len(MAP_FAMILIES['softplus']['param_range'])//2]

    lam_a, gate_a = get_gate_grid('scaled_exp', cmb_alpha)
    lam_b, gate_b = get_gate_grid('softplus', dl_k)

    # Align grids
    common_lam = np.intersect1d(
        np.round(lam_a, 4),
        np.round(lam_b, 4)
    )

    # Build aligned arrays
    def align(lam_vals, gate_grid):
        result = {}
        for l, g in zip(np.round(lam_vals, 4), gate_grid):
            result[l] = g
        return np.array([result.get(l, np.nan) for l in common_lam])

    ga = align(lam_a, gate_a)
    gb = align(lam_b, gate_b)
    diff = ga - gb  # +1 = A passes B fails, -1 = B passes A fails, 0 = agree

    for ax, gate_data, title, color in zip(
        axes,
        [ga, gb, diff],
        [f'Frame A: CMB-like\n(scaled_exp, α={cmb_alpha:.2f})',
         f'Frame B: Distance-ladder-like\n(softplus, k={dl_k:.2f})',
         'Frame Disagreement\n(A-B: +1=A only, -1=B only, 0=agree)'],
        ['Blues', 'Oranges', 'RdBu']
    ):
        valid = ~np.isnan(gate_data)
        if valid.any():
            ax.bar(common_lam[valid], gate_data[valid],
                   width=0.05, color=plt.cm.get_cmap(color)(0.6),
                   edgecolor='white', linewidth=0.5)
        ax.set_xlabel(r'Memory $\lambda_{mem}$', fontsize=11)
        ax.set_ylabel('Gate score / difference', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.3)

        # Mark Hubble-analogous zone
        ax.axvspan(0.6, 0.85, alpha=0.1, color='green',
                   label='Hubble-analogous\nmemory regime')
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_metric_profiles(df, family_name='scaled_exp', sigma_fixed=0.7,
                          lam_fixed=0.8, save_path=None):
    """
    Profile plot: how each gate metric varies with map parameter,
    at fixed (lam_mem, sigma). Shows which criterion breaks first.
    """
    family = MAP_FAMILIES[family_name]
    param_name = family['param_name']

    sub = df[(df['family'] == family_name) &
             (np.abs(df['sigma'] - sigma_fixed) < 0.05) &
             (np.abs(df['lam_mem'] - lam_fixed) < 0.04)].sort_values(param_name)

    if sub.empty:
        print(f"No data for profile plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Gate Metric Profiles — {family_name}\n'
                 f'(λ_mem={lam_fixed:.2f}, σ={sigma_fixed})', fontsize=12)

    params = sub[param_name].values
    metrics = ['beta', 'entropy_slope', 'entropy_r2', 'curvature_persistence']
    gates   = ['gate_beta', 'gate_entropy_slope', 'gate_entropy_r2', 'gate_curvature']
    titles  = ['Tail exponent β\n(gate: 3.0–5.5)',
               'Entropy slope\n(gate: < 0)',
               'Entropy R²\n(gate: > 0.35)',
               'Curvature persistence\n(gate: > 0.30)']
    thresholds = [(3.0, 5.5), (None, 0.0), (0.35, None), (0.30, None)]

    for ax, metric, gate, title, thresh in zip(
            axes.flat, metrics, gates, titles, thresholds):

        vals = sub[metric].values
        gate_vals = sub[gate].values.astype(float)

        # Color points by pass/fail
        colors_pts = ['#2ca02c' if g >= 0.5 else '#d62728' for g in gate_vals]
        ax.scatter(params, vals, c=colors_pts, s=40, zorder=5)
        ax.plot(params, vals, 'k-', alpha=0.3, lw=1)

        # Threshold lines
        lo, hi = thresh
        if lo is not None:
            ax.axhline(lo, color='blue', ls='--', alpha=0.6, label=f'threshold = {lo}')
        if hi is not None:
            ax.axhline(hi, color='blue', ls='--', alpha=0.6, label=f'threshold = {hi}')

        ax.set_xlabel(family['xlabel'], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Shade the passing region
        ax.axvspan(params[gate_vals >= 0.5].min() if (gate_vals >= 0.5).any() else params[-1],
                   params.max(), alpha=0.06, color='green')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# SECTION 7: MAIN EXECUTION
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    OUT_DIR = '/home/claude/pmir_phase_output'
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PMIR OBSERVATION MAP PHASE DIAGRAM SWEEP")
    print("=" * 70)
    print(f"Structural Gate: 3.0 ≤ β ≤ 5.5  |  entropy_slope < 0")
    print(f"                 entropy_R² > 0.35  |  curvature_pers > 0.30")
    print(f"Grid: {len(LAM_MEM_VALUES)} × {len(SIGMA_VALUES)} latent points per map param")
    print()

    # Run full sweep — use smaller T and fewer seeds for initial run
    # Increase T=6000, n_seeds=5 for publication-quality results
    df = run_full_sweep(
        families=['scaled_exp', 'softplus', 'powerlaw', 'sigmoid_scaled', 'boxcox'],
        T=3000,
        n_seeds=3
    )

    # Save full results
    csv_path = os.path.join(OUT_DIR, 'pmir_phase_diagram_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved: {csv_path}")
    print(f"Total points: {len(df)}")
    print(f"Overall gate pass rate: {100*df['gate_pass'].mean():.1f}%")
    print()

    # Summary by family
    print("Gate pass rate by map family:")
    print("-" * 50)
    for fname in df['family'].unique():
        sub = df[df['family'] == fname]
        rate = 100 * sub['gate_pass'].mean()
        print(f"  {fname:20s}: {rate:5.1f}% ({sub['gate_pass'].sum()}/{len(sub)} points)")

    print("\nGenerating phase diagram plots...")

    # Individual family phase diagrams
    for fname in MAP_FAMILIES.keys():
        plot_phase_diagram_single_family(
            df, fname, sigma_fixed=0.7,
            save_path=os.path.join(OUT_DIR, f'pmir_phase_{fname}.png')
        )

    # Metric profiles for scaled_exp (key map)
    plot_metric_profiles(
        df, family_name='scaled_exp', sigma_fixed=0.7, lam_fixed=0.8,
        save_path=os.path.join(OUT_DIR, 'pmir_metric_profiles_scaled_exp.png')
    )

    # Composite boundary
    boundary_data = plot_composite_boundary(
        df, sigma_fixed=0.7,
        save_path=os.path.join(OUT_DIR, 'pmir_phase_composite_boundary.png')
    )

    # Hubble tension frame comparison
    plot_hubble_frame_comparison(
        df,
        save_path=os.path.join(OUT_DIR, 'pmir_hubble_frame_comparison.png')
    )

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print("Files generated:")
    for f in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:50s} {size_kb:7.1f} KB")
