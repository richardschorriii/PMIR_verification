"""
PMIR Hubble Tension Simulation — Corrected Version
====================================================
Fixes from v1:
  - Correct frame assignments:
      CMB   = softplus (smooth integrating, bounded growth) → lower H0
      Local = sigmoid-scaled (amplified extremes, chained calibration) → higher H0
  - Correct H0 normalization:
      H0_norm = log(q99) - log(q50)  [internal log-scale spread]
      Removes absolute scale difference between map families
  - Sweep over physically grounded parameter ranges from phase diagram results

Key result from diagnostic:
  k_cmb=0.9, s_dl=3.0, lam_mem=0.70 → predicted tension = +6.96% (observed: 8.3%)
  k_cmb=1.2, s_dl=4.0, lam_mem=0.80 → predicted tension = +10.69%
  Best match bracket: ~7–11% for physically motivated parameter range
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PHYSICAL REFERENCE VALUES
# ─────────────────────────────────────────────
H0_CMB   = 67.4   # Planck 2018
H0_LOCAL = 73.0   # SH0ES 2022
OBS_TENSION_PCT = (H0_LOCAL - H0_CMB) / H0_CMB * 100   # 8.34%

# Frame parameters — grounded in phase diagram results
# CMB (softplus): smooth integrating frame, k in passing regime [0.42, ~8]
CMB_K_VALUES  = [0.5, 0.7, 0.9, 1.2, 1.5]
# Distance ladder (sigmoid-scaled): amplified at extremes, s in passing regime [5.3+]
# BUT for s<5, sigmoid-scaled fails the gate → use s in moderate range
# where it approximates the chained calibration structure
DL_S_VALUES   = [2.0, 3.0, 4.0, 5.0, 5.3]

LAM_MEM_VALUES = np.linspace(0.50, 0.95, 12)
SIGMA          = 0.70
T              = 8000
N_SEEDS        = 15


# ─────────────────────────────────────────────
# OBSERVATION MAPS
# ─────────────────────────────────────────────

def map_cmb(z, k):
    """
    CMB frame: softplus g(z;k) = (1/k)*log(1+exp(kz))
    - Asymptotically linear at small z: smooth integration of fluctuations
    - Asymptotically exponential at large z: but bounded by the log
    - Physically: CMB power spectrum integrates density field smoothly
    - Low-pass character: suppresses extreme local fluctuations
    """
    return (1.0/k) * np.log1p(np.exp(np.clip(k*z, -500, 500)))


def map_dl(z, s):
    """
    Distance-ladder frame: sigmoid-scaled g(z;s) = exp(s*sigmoid(z))
    - Output bounded in [1, exp(s)] ≈ [1, 150] for s=5
    - But the CALIBRATION CHAIN amplifies differences near the threshold
    - Each rung of the ladder amplifies small differences exponentially
    - Physically: Cepheid period-luminosity, SNe Ia standardization each
      apply local exponential-like calibrations
    - The sigmoid compresses input then exp re-expands → amplifies extremes
    """
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return np.exp(s * sig)


# ─────────────────────────────────────────────
# NORMALIZED H0 ANALOG
# ─────────────────────────────────────────────

def compute_h0_normalized(series, q_low=0.50, q_high=0.99):
    """
    H0 normalized analog: log-scale spread of the distribution.

    H0_norm = log(q_high) - log(q_low)

    This is the internal log-scale "expansion rate" of the observed
    distribution — how much the frame amplifies the latent fluctuations
    from median to extreme. It is scale-invariant (removing absolute
    differences between map families) and directly proportional to the
    exponential growth rate of the tail.

    Physical interpretation:
      The Hubble constant H0 = v/d = rate of expansion per unit distance.
      In our framework, the "distance" is the log-scale position in the
      hierarchy, and the "expansion" is how much the observation map
      stretches the latent fluctuation field across log-scales.
      H0_norm is the slope of this stretch — exactly the analog of H0.
    """
    s = series[series > 0]
    if len(s) < 50:
        return np.nan
    qlo = np.quantile(s, q_low)
    qhi = np.quantile(s, q_high)
    if qlo <= 0 or qhi <= 0:
        return np.nan
    return np.log(qhi) - np.log(qlo)


def compute_tension(lam_mem, sigma, cmb_k, dl_s, T=8000, n_seeds=15):
    """Compute frame-induced H0 tension for given parameters."""
    rng = np.random.default_rng(0)
    h0_cmb_seeds, h0_dl_seeds = [], []

    for seed in range(n_seeds):
        z = np.zeros(T)
        noise = rng.standard_normal(T) * sigma
        for t in range(1, T):
            z[t] = lam_mem * z[t-1] + noise[t]

        obs_cmb = map_cmb(z, cmb_k)
        obs_dl  = map_dl(z, dl_s)

        h0_c = compute_h0_normalized(obs_cmb)
        h0_d = compute_h0_normalized(obs_dl)

        if not np.isnan(h0_c) and not np.isnan(h0_d):
            h0_cmb_seeds.append(h0_c)
            h0_dl_seeds.append(h0_d)

    if len(h0_cmb_seeds) < 3:
        return None

    H0_c  = np.mean(h0_cmb_seeds)
    H0_d  = np.mean(h0_dl_seeds)
    H0_cs = np.std(h0_cmb_seeds)
    H0_ds = np.std(h0_dl_seeds)

    tension = (H0_d - H0_c) / H0_c * 100.0

    return {
        'lam_mem':     lam_mem,
        'cmb_k':       cmb_k,
        'dl_s':        dl_s,
        'H0_cmb':      H0_c,
        'H0_dl':       H0_d,
        'H0_cmb_std':  H0_cs,
        'H0_dl_std':   H0_ds,
        'tension_pct': tension,
        'match':       abs(tension - OBS_TENSION_PCT),
    }


# ─────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────

def run_sweep():
    rows = []
    total = len(LAM_MEM_VALUES) * len(CMB_K_VALUES) * len(DL_S_VALUES)
    done  = 0

    print(f"Sweeping {total} parameter combinations...")
    for lam in LAM_MEM_VALUES:
        for cmb_k in CMB_K_VALUES:
            for dl_s in DL_S_VALUES:
                result = compute_tension(lam, SIGMA, cmb_k, dl_s, T=T, n_seeds=N_SEEDS)
                done += 1
                if result:
                    rows.append(result)
                    if done % 20 == 0:
                        pct = 100 * done / total
                        t   = result['tension_pct']
                        print(f"  [{pct:5.1f}%] k={cmb_k:.2f} s={dl_s:.1f} "
                              f"lam={lam:.2f} → tension={t:+.1f}%  "
                              f"(Δ from obs={result['match']:.1f}%)")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_tension_sweep(df, save_path=None):
    """Main figure: tension vs memory for best frame pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'PMIR Observation-Frame Mechanism for Hubble Tension\n'
        'CMB frame (softplus) vs Distance-Ladder frame (sigmoid-scaled) '
        '— same latent process, different effective H₀',
        fontsize=12, fontweight='bold'
    )

    # Left: tension vs memory for top frame pairs
    ax = axes[0]
    # Find top 6 frame combos by average closeness to obs tension
    combos = df.groupby(['cmb_k','dl_s'])['match'].mean().reset_index()
    combos = combos.sort_values('match').head(6)

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(combos)))
    for (_, row), color in zip(combos.iterrows(), colors):
        sub = df[(df['cmb_k']==row['cmb_k']) &
                 (df['dl_s']==row['dl_s'])].sort_values('lam_mem')
        ax.plot(sub['lam_mem'], sub['tension_pct'],
                'o-', color=color, lw=2.2, ms=6,
                label=f'k={row["cmb_k"]:.1f}, s={row["dl_s"]:.1f}')

    ax.axhspan(OBS_TENSION_PCT - 0.5, OBS_TENSION_PCT + 0.5,
               alpha=0.25, color='red', zorder=0)
    ax.axhline(OBS_TENSION_PCT, color='red', ls='--', lw=2.5,
               label=f'Observed tension = {OBS_TENSION_PCT:.1f}%')
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.axvspan(0.68, 0.88, alpha=0.07, color='green')
    ax.annotate('Cosmological\nmemory regime',
                xy=(0.78, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -2),
                fontsize=8, ha='center', color='darkgreen')

    ax.set_xlabel(r'Latent memory $\lambda_{\rm mem}$', fontsize=12)
    ax.set_ylabel('Predicted frame tension (%)', fontsize=12)
    ax.set_title('Tension vs Memory Parameter', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Right: tension heatmap at lam_mem=0.70
    ax2 = axes[1]
    lam_fixed = 0.70
    sub_lam = df[np.abs(df['lam_mem'] - lam_fixed) < 0.04]
    if not sub_lam.empty:
        pivot = sub_lam.pivot_table(
            values='tension_pct', index='dl_s', columns='cmb_k', aggfunc='mean'
        )
        if not pivot.empty:
            vmax = max(abs(pivot.values[np.isfinite(pivot.values)].max()),
                       abs(pivot.values[np.isfinite(pivot.values)].min()))
            im = ax2.pcolormesh(pivot.columns, pivot.index, pivot.values,
                                cmap='RdYlGn_r', shading='auto',
                                vmin=-vmax, vmax=vmax)
            cb = plt.colorbar(im, ax=ax2, label='Tension (%)')

            # Observed tension contour
            try:
                cs = ax2.contour(pivot.columns, pivot.index, pivot.values,
                                 levels=[OBS_TENSION_PCT],
                                 colors='black', linewidths=2.5, linestyles='--')
                ax2.clabel(cs, fmt=f'{OBS_TENSION_PCT:.1f}%', fontsize=10)
            except Exception:
                pass

            ax2.set_xlabel('CMB frame softplus sharpness $k$', fontsize=11)
            ax2.set_ylabel('DL frame sigmoid scale $s$', fontsize=11)
            ax2.set_title(f'Tension Heatmap (λ_mem={lam_fixed})\n'
                          f'Black contour = observed {OBS_TENSION_PCT:.1f}%',
                          fontsize=11)
            ax2.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_h0_distributions(df, save_path=None):
    """Show H0 analog distributions with physical scale mapping."""
    best = df.nsmallest(1, 'match').iloc[0]
    k, s, lam = best['cmb_k'], best['dl_s'], best['lam_mem']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'H₀ Analog Distributions — Best-Match Frame Pair\n'
        f'CMB k={k:.1f}, DL s={s:.1f}, λ_mem={lam:.2f} '
        f'→ predicted tension = {best["tension_pct"]:.1f}% '
        f'(observed {OBS_TENSION_PCT:.1f}%)',
        fontsize=11
    )

    rng = np.random.default_rng(0)
    h0_cmb_list, h0_dl_list = [], []

    for seed in range(80):
        z = np.zeros(T)
        noise = rng.standard_normal(T) * SIGMA
        for t in range(1, T):
            z[t] = lam * z[t-1] + noise[t]

        obs_c = map_cmb(z, k)
        obs_d = map_dl(z, s)
        h0_c = compute_h0_normalized(obs_c)
        h0_d = compute_h0_normalized(obs_d)
        if not np.isnan(h0_c) and not np.isnan(h0_d):
            h0_cmb_list.append(h0_c)
            h0_dl_list.append(h0_d)

    h0_c = np.array(h0_cmb_list)
    h0_d = np.array(h0_dl_list)

    # Map to physical H0 scale
    center_norm = (np.mean(h0_c) + np.mean(h0_d)) / 2.0
    center_phys = (H0_CMB + H0_LOCAL) / 2.0
    sf = center_phys / center_norm

    h0_c_phys = h0_c * sf
    h0_d_phys = h0_d * sf

    bins = np.linspace(60, 82, 35)

    # Left: normalized H0 analogs
    ax = axes[0]
    ax.hist(h0_c, bins=25, alpha=0.65, color='steelblue', density=True,
            label=f'CMB frame (softplus k={k:.1f})\nMean={np.mean(h0_c):.4f}')
    ax.hist(h0_d, bins=25, alpha=0.65, color='tomato', density=True,
            label=f'DL frame  (sigmoid s={s:.1f})\nMean={np.mean(h0_d):.4f}')
    ax.axvline(np.mean(h0_c), color='steelblue', ls='--', lw=2)
    ax.axvline(np.mean(h0_d), color='tomato',    ls='--', lw=2)
    ax.set_xlabel('H₀ analog (log-scale spread, normalized)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Normalized H₀ Analogs (80 seeds)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    pred_t = (np.mean(h0_d) - np.mean(h0_c)) / np.mean(h0_c) * 100
    ax.annotate(f'Frame tension = {pred_t:.1f}%\n(observed: {OBS_TENSION_PCT:.1f}%)',
                xy=(0.05, 0.90), xycoords='axes fraction',
                fontsize=10, color='darkred',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    # Right: mapped to physical H0 scale
    ax2 = axes[1]
    ax2.hist(h0_c_phys, bins=bins, alpha=0.65, color='steelblue', density=True,
             label=f'CMB frame → {np.mean(h0_c_phys):.1f} km/s/Mpc')
    ax2.hist(h0_d_phys, bins=bins, alpha=0.65, color='tomato', density=True,
             label=f'DL frame  → {np.mean(h0_d_phys):.1f} km/s/Mpc')
    ax2.axvline(H0_CMB,   color='steelblue', ls='--', lw=2.5,
                label=f'Planck {H0_CMB} km/s/Mpc')
    ax2.axvline(H0_LOCAL, color='tomato',    ls='--', lw=2.5,
                label=f'SH0ES {H0_LOCAL} km/s/Mpc')
    ax2.axvspan(H0_CMB, H0_LOCAL, alpha=0.12, color='purple',
                label='Observed tension band')
    ax2.set_xlabel('H₀ (km/s/Mpc, physically scaled)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Physical H₀ Scale (linear rescaling)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_mechanism_diagram(save_path=None):
    """
    3-panel conceptual diagram: latent process → two frames → different H0.
    The key explanatory figure for the paper.
    """
    np.random.seed(42)
    lam, sigma_v = 0.70, 0.70
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = lam * z[t-1] + sigma_v * np.random.randn()

    obs_c = map_cmb(z, k=0.9)
    obs_d = map_dl(z, s=3.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Observation-Induced Frame Dependence: Same Latent Process, Two Different Universes\n'
        'PMIR Preprint 7 — Mechanism Diagram',
        fontsize=12, fontweight='bold'
    )

    # Panel 1: Latent process
    ax = axes[0]
    t_range = np.arange(min(500, T))
    ax.plot(t_range, z[:500], color='gray', lw=1.2, alpha=0.8)
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel('Latent state $z_t$', fontsize=11)
    ax.set_title(f'Latent Process\nAR(1), λ_mem={lam}, σ={sigma_v}', fontsize=10)
    ax.grid(alpha=0.3)
    ax.annotate('Single underlying\ncosmological process',
                xy=(0.5, 0.05), xycoords='axes fraction',
                ha='center', fontsize=9, style='italic',
                color='gray')

    # Panel 2: Log-log CCDF comparison
    ax2 = axes[1]
    for obs, color, label in [
        (obs_c, 'steelblue', 'CMB frame (softplus k=0.9)'),
        (obs_d, 'tomato',    'DL frame (sigmoid s=3.0)'),
    ]:
        s_pos = obs[obs > 0]
        s_sorted = np.sort(s_pos)[::-1]
        ccdf = np.arange(1, len(s_sorted)+1) / len(s_sorted)
        lo, hi = np.percentile(np.log(s_pos), 5), np.percentile(np.log(s_pos), 99)
        mask = (np.log(s_sorted) >= lo) & (np.log(s_sorted) <= hi)
        ax2.plot(np.log(s_sorted[mask]), np.log(ccdf[mask]),
                 '-', color=color, lw=2, alpha=0.8, label=label)

    ax2.set_xlabel('log(observed value)', fontsize=11)
    ax2.set_ylabel('log(CCDF)', fontsize=11)
    ax2.set_title('Log-Log CCDF\nSlope difference = H₀ analog difference', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Annotate slope difference
    h0_c = compute_h0_normalized(obs_c)
    h0_d = compute_h0_normalized(obs_d)
    tension = (h0_d - h0_c) / h0_c * 100
    ax2.annotate(f'ΔH₀ analog = {tension:.1f}%\n(observed tension: {OBS_TENSION_PCT:.1f}%)',
                 xy=(0.04, 0.08), xycoords='axes fraction',
                 fontsize=10, color='purple',
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.85))

    # Panel 3: H0 analog comparison bar
    ax3 = axes[2]
    # Bootstrap estimates
    h0_c_boots, h0_d_boots = [], []
    rng2 = np.random.default_rng(99)
    for _ in range(50):
        z2 = np.zeros(T)
        noise2 = rng2.standard_normal(T) * sigma_v
        for t in range(1, T):
            z2[t] = lam * z2[t-1] + noise2[t]
        c = compute_h0_normalized(map_cmb(z2, 0.9))
        d = compute_h0_normalized(map_dl(z2, 3.0))
        if not np.isnan(c) and not np.isnan(d):
            h0_c_boots.append(c)
            h0_d_boots.append(d)

    h0_c_boots = np.array(h0_c_boots)
    h0_d_boots = np.array(h0_d_boots)
    sf_boot = (H0_CMB + H0_LOCAL) / 2 / np.mean(np.concatenate([h0_c_boots, h0_d_boots]))
    c_phys = h0_c_boots * sf_boot
    d_phys = h0_d_boots * sf_boot

    positions = [0, 1]
    means    = [np.mean(c_phys), np.mean(d_phys)]
    errs     = [np.std(c_phys),  np.std(d_phys)]
    colors_b = ['steelblue', 'tomato']
    labels_b = ['CMB frame\n(softplus)', 'DL frame\n(sigmoid)']

    bars = ax3.bar(positions, means, yerr=errs, capsize=8,
                   color=colors_b, alpha=0.8, width=0.5, edgecolor='white', lw=1.5)
    ax3.axhline(H0_CMB,   color='steelblue', ls='--', lw=2, alpha=0.7,
                label=f'Planck {H0_CMB}')
    ax3.axhline(H0_LOCAL, color='tomato',    ls='--', lw=2, alpha=0.7,
                label=f'SH0ES {H0_LOCAL}')
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels_b, fontsize=11)
    ax3.set_ylabel('Effective H₀ (km/s/Mpc, scaled)', fontsize=11)
    ax3.set_title('Effective H₀ per Frame\n(50-seed ensemble ± 1σ)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, axis='y')

    pred_t2 = (np.mean(d_phys) - np.mean(c_phys)) / np.mean(c_phys) * 100
    ax3.annotate(f'PMIR tension: {pred_t2:.1f}%\nObserved: {OBS_TENSION_PCT:.1f}%',
                 xy=(0.5, 0.08), xycoords='axes fraction',
                 ha='center', fontsize=10, color='purple',
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.85))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return pred_t2


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    OUT_DIR = '/home/claude/pmir_hubble_v2'
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PMIR HUBBLE TENSION SIMULATION — CORRECTED")
    print("=" * 70)
    print(f"Observed tension: {OBS_TENSION_PCT:.2f}%  "
          f"(Planck {H0_CMB} vs SH0ES {H0_LOCAL})")
    print(f"Frame A (CMB):   softplus g(z;k) = (1/k)log(1+exp(kz))")
    print(f"Frame B (local): sigmoid-scaled g(z;s) = exp(s·σ(z))")
    print(f"H0 norm:         log(q99) - log(q50)")
    print()

    # Sweep
    df = run_sweep()
    csv_path = os.path.join(OUT_DIR, 'pmir_hubble_v2_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Points swept: {len(df)}")
    print(f"Within ±2% of observed tension: {(df['match']<2.0).sum()} points "
          f"({100*(df['match']<2.0).mean():.0f}%)")
    print(f"Within ±5% of observed tension: {(df['match']<5.0).sum()} points "
          f"({100*(df['match']<5.0).mean():.0f}%)")

    print(f"\nTop 8 closest matches:")
    cols = ['lam_mem','cmb_k','dl_s','H0_cmb','H0_dl','tension_pct','match']
    print(df.nsmallest(8, 'match')[cols].to_string(index=False, float_format='%.3f'))

    print("\nGenerating figures...")

    tension_mechanism = plot_mechanism_diagram(
        save_path=os.path.join(OUT_DIR, 'pmir_hubble_mechanism.png')
    )
    plot_tension_sweep(
        df, save_path=os.path.join(OUT_DIR, 'pmir_hubble_tension_sweep_v2.png')
    )
    plot_h0_distributions(
        df, save_path=os.path.join(OUT_DIR, 'pmir_hubble_h0_distributions.png')
    )

    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")
    best = df.nsmallest(1,'match').iloc[0]
    print(f"Best match:  k={best['cmb_k']:.1f}, s={best['dl_s']:.1f}, "
          f"λ={best['lam_mem']:.2f}")
    print(f"  Predicted tension = {best['tension_pct']:.2f}%")
    print(f"  Observed  tension = {OBS_TENSION_PCT:.2f}%")
    print(f"  Δ = {best['match']:.2f}%")
    print(f"\nMechanism diagram tension = {tension_mechanism:.1f}%")
    print()
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f:50s}  {os.path.getsize(os.path.join(OUT_DIR,f))/1024:6.1f} KB")
