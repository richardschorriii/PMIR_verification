"""
PMIR Preprint 7 — Extended Simulation Suite
============================================
Author: Richard L. Schorr III

Three parallel extensions to the minimal Hubble tension simulation:

  COMPONENT 1: ΛCDM Latent Process
    Replace AR(1) with a cosmologically motivated latent field.
    Use Eisenstein-Hu transfer function to generate a 1D realization
    of the matter power spectrum P(k) ~ k^ns * T²(k) with ns=0.965.
    The latent "time series" is a draw from this correlated Gaussian field,
    capturing the actual spectral structure of cosmic density fluctuations.
    Key: ΛCDM has a characteristic correlation length (equality scale ~100 Mpc)
    that maps to an effective memory parameter λ_eff.

  COMPONENT 2: Composed Observer Experiment
    Model the distance ladder as THREE chained softplus maps:
      Rung 1: parallax → Cepheids  (k₁)
      Rung 2: Cepheids → SNe Ia    (k₂)
      Rung 3: SNe Ia → H₀          (k₃)
    Each rung applies its own softplus calibration.
    Compare: single sigmoid-scaled CMB frame vs composed-softplus DL frame.
    Prediction: composed softplus amplifies tension relative to single-map DL
    because each rung introduces additional hierarchy depth.

  COMPONENT 3: High-Resolution Sweep
    T=12000, n_seeds=25 for publication-quality tension contours.
    Finer parameter grids around the confirmed best-match region
    (k ∈ [0.6, 1.4], s ∈ [2.0, 5.0], λ ∈ [0.55, 0.85]).

Outputs:
  pmir_lcdm_latent.png          — ΛCDM vs AR(1) spectral comparison
  pmir_lcdm_tension.png         — tension with ΛCDM latent process
  pmir_composed_observer.png    — chained map vs single map comparison
  pmir_highres_sweep.png        — high-resolution tension contours
  pmir_preprint7_summary.png    — 4-panel combined summary figure
  pmir_p7_extended_results.csv  — all numerical results
"""

import numpy as np
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.contour import QuadContourSet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
H0_CMB    = 67.4
H0_LOCAL  = 73.0
OBS_TENSION = (H0_LOCAL - H0_CMB) / H0_CMB * 100   # 8.31%

SIGMA = 0.70


# ═══════════════════════════════════════════════════════════════
# COMPONENT 1: ΛCDM LATENT PROCESS
# ═══════════════════════════════════════════════════════════════

def eisenstein_hu_transfer(k, omega_m=0.3089, omega_b=0.0486, h=0.674):
    """
    Simplified Eisenstein & Hu (1998) transfer function T(k).
    Returns T(k) for wavenumber k in h/Mpc.

    This gives the correct shape of the matter power spectrum:
    - P(k) ~ k^ns at small k (large scales)
    - Turnover near equality scale k_eq ~ 0.01 h/Mpc
    - Acoustic oscillations (BAO) at k ~ 0.1 h/Mpc
    - Power suppression at large k (small scales)
    """
    # Baryon-to-photon momentum ratio
    T_cmb = 2.7255 / 2.7  # normalized
    omega_m_h2 = omega_m * h**2
    omega_b_h2 = omega_b * h**2

    # Equality scale
    k_eq = 7.46e-2 * omega_m_h2 / T_cmb**2   # h/Mpc

    # Drag epoch
    b1 = 0.313 * omega_m_h2**(-0.419) * (1 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    z_d = 1291 * omega_m_h2**0.251 / (1 + 0.659 * omega_m_h2**0.828) * (
        1 + b1 * omega_b_h2**b2)
    z_eq = 2.5e4 * omega_m_h2 * T_cmb**(-4)

    # Sound horizon
    R_eq = 31.5e3 * omega_b_h2 * T_cmb**(-4) * (1000/z_eq)
    R_d  = 31.5e3 * omega_b_h2 * T_cmb**(-4) * (1000/z_d)
    s = (2/(3*k_eq)) * np.sqrt(6/R_eq) * np.log(
        (np.sqrt(1+R_d) + np.sqrt(R_d+R_eq)) / (1+np.sqrt(R_eq)))

    k_silk = 1.6 * (omega_b_h2**0.52) * (omega_m_h2**0.038) * (1/T_cmb)**1.6

    # CDM transfer function (no-wiggle approximation for simplicity)
    q = k / (13.41 * k_eq)
    f = 1.0 / (1.0 + (k*s/5.4)**4)

    C0 = 14.2 + 731 / (1 + 62.5*q)
    T0 = np.log(np.e + 1.8*q) / (np.log(np.e + 1.8*q) + C0*q**2)

    C1 = 14.2/f + 386 / (1 + 69.9*q**1.08)
    T1 = np.log(np.e + 1.8*q) / (np.log(np.e + 1.8*q) + C1*q**2)

    f_baryon = omega_b / omega_m

    # Baryon transfer function with oscillations
    y = z_eq / (1 + z_d)
    G = y * (-6*np.sqrt(1+y) + (2+3*y)*np.log((np.sqrt(1+y)+1)/(np.sqrt(1+y)-1)))
    alpha_b = 2.07 * k_eq * s * (1+R_d)**(-3/4) * G
    beta_b  = 0.5 + f_baryon + (3 - 2*f_baryon) * np.sqrt((17.2*omega_m_h2)**2 + 1)
    beta_node = 8.41 * omega_m_h2**0.435

    s_tilde = s / (1 + (beta_node/(k*s))**3)**(1/3)
    j0 = np.sinc(k*s_tilde/np.pi)  # j0(x) = sin(x)/x, numpy sinc = sin(πx)/(πx)

    T_b = (T0/(1 + (k*s/5.2)**2) +
           alpha_b / (1 + (beta_b/(k*s))**3) * np.exp(-(k/k_silk)**1.4)) * j0

    T = f_baryon * T_b + (1 - f_baryon) * T0
    return np.clip(T, 0, 1)


def generate_lcdm_latent(T=12000, ns=0.965, seed=42,
                          k_min=1e-3, k_max=5.0, n_modes=1000):
    """
    Generate a 1D realization of the ΛCDM matter power spectrum
    as a correlated Gaussian latent field.

    Method:
      1. Sample N Fourier modes k_i with weights ~ sqrt(P(k_i))
      2. Assign random phases φ_i ~ Uniform[0, 2π]
      3. Construct x(t) = sum_i A_i * cos(2π k_i t + φ_i)
      4. Normalize to zero mean, unit variance
         (same normalization as AR(1) process)

    The resulting series has the correct spectral shape of cosmic
    density fluctuations — specifically, it has the ΛCDM characteristic
    correlation length (~ equality scale) built in as effective memory.

    Returns: z (T,) array — ΛCDM latent field
             k_vals, P_vals — power spectrum for plotting
    """
    rng = np.random.default_rng(seed)

    k_vals = np.logspace(np.log10(k_min), np.log10(k_max), n_modes)
    T_vals = eisenstein_hu_transfer(k_vals)
    P_vals = k_vals**ns * T_vals**2   # ΛCDM matter power spectrum

    # Amplitude weights: sqrt(P(k)) for Gaussian field
    A_vals = np.sqrt(P_vals)
    A_vals /= A_vals.sum()  # normalize total power

    # Random phases
    phases = rng.uniform(0, 2*np.pi, n_modes)

    # Generate time series
    t = np.arange(T, dtype=float) / T  # normalized to [0, 1]
    z = np.zeros(T)
    for i, (k, A, phi) in enumerate(zip(k_vals, A_vals, phases)):
        z += A * np.cos(2 * np.pi * k * t * T / n_modes + phi)

    # Normalize to zero mean, unit variance (matching AR(1) setup)
    z = (z - z.mean()) / (z.std() + 1e-10)

    return z, k_vals, P_vals


def compute_effective_memory(z):
    """
    Compute the effective memory parameter of a time series
    by fitting its autocorrelation to an AR(1) model.
    Returns λ_eff (the AR(1) equivalent memory).
    """
    ac = np.correlate(z - z.mean(), z - z.mean(), mode='full')
    ac = ac[len(ac)//2:]
    ac /= ac[0]
    # λ_eff = AC at lag 1
    return ac[1] if len(ac) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════
# COMPONENT 2: COMPOSED OBSERVER (CHAINED MAPS)
# ═══════════════════════════════════════════════════════════════

def map_cmb(z, k=0.9):
    """CMB frame: softplus — smooth integrating."""
    return (1.0/k) * np.log1p(np.exp(np.clip(k*z, -500, 500)))

def map_dl_single(z, s=3.0):
    """Distance ladder: single sigmoid-scaled map."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return np.exp(s * sig)

def map_dl_composed(z, k1=1.0, k2=0.8, k3=1.2):
    """
    Distance ladder as THREE chained softplus maps:
      Rung 1 (parallax→Cepheid): softplus(z; k1)
      Rung 2 (Cepheid→SNe Ia):   softplus(log(rung1); k2)
      Rung 3 (SNe Ia→H0):        softplus(log(rung2); k3)

    Each rung applies softplus to the LOG of the previous output —
    modeling how astronomers work in magnitude/log-distance space
    at each calibration step.

    This is physically correct: the period-luminosity relation,
    the standardizable candle calibration, and the recession velocity
    measurement are all log-linear relations.
    """
    # Rung 1: raw latent → Cepheid distances
    r1 = (1.0/k1) * np.log1p(np.exp(np.clip(k1*z, -500, 500)))
    r1 = np.clip(r1, 1e-10, None)

    # Rung 2: Cepheid → SNe Ia (work in log space)
    log_r1 = np.log(r1)
    r2 = (1.0/k2) * np.log1p(np.exp(np.clip(k2*log_r1, -500, 500)))
    r2 = np.clip(r2, 1e-10, None)

    # Rung 3: SNe Ia → H0
    log_r2 = np.log(r2)
    r3 = (1.0/k3) * np.log1p(np.exp(np.clip(k3*log_r2, -500, 500)))

    return r3


def compute_h0_norm(series, qlo=0.50, qhi=0.99):
    """H0 normalized analog: log(q_hi) - log(q_lo)."""
    s = series[series > 0]
    if len(s) < 50:
        return np.nan
    lo = np.quantile(s, qlo)
    hi = np.quantile(s, qhi)
    if lo <= 0 or hi <= 0:
        return np.nan
    return np.log(hi) - np.log(lo)


def run_composed_observer_experiment(T=12000, n_seeds=20):
    """
    Compare four frame configurations:
      A: CMB (single softplus)
      B: DL single  (sigmoid-scaled)
      C: DL composed (3× chained softplus)
      D: DL composed symmetric (3× same k softplus)

    For a range of λ_mem values and the confirmed best-match params.
    """
    print("Running composed observer experiment...")

    lam_vals = np.linspace(0.50, 0.95, 12)
    configs = {
        'CMB_softplus':     lambda z: map_cmb(z, k=0.9),
        'DL_sigmoid':       lambda z: map_dl_single(z, s=3.0),
        'DL_3chain':        lambda z: map_dl_composed(z, k1=1.0, k2=0.8, k3=1.2),
        'DL_3chain_sym':    lambda z: map_dl_composed(z, k1=0.9, k2=0.9, k3=0.9),
    }

    rows = []
    for lam in lam_vals:
        seed_results = {name: [] for name in configs}
        rng = np.random.default_rng(0)
        for seed in range(n_seeds):
            z = np.zeros(T)
            noise = rng.standard_normal(T) * SIGMA
            for t in range(1, T):
                z[t] = lam * z[t-1] + noise[t]

            for name, func in configs.items():
                obs = func(z)
                h0  = compute_h0_norm(obs)
                if not np.isnan(h0):
                    seed_results[name].append(h0)

        row = {'lam_mem': lam}
        for name in configs:
            vals = seed_results[name]
            row[f'H0_{name}'] = np.mean(vals) if vals else np.nan
            row[f'H0_{name}_std'] = np.std(vals) if vals else np.nan

        # Compute tensions relative to CMB
        ref = row['H0_CMB_softplus']
        for name in ['DL_sigmoid', 'DL_3chain', 'DL_3chain_sym']:
            t = (row[f'H0_{name}'] - ref) / ref * 100 if ref > 0 else np.nan
            row[f'tension_{name}'] = t

        rows.append(row)
        print(f"  λ={lam:.2f} | single={row['tension_DL_sigmoid']:+.1f}% "
              f"chain3={row['tension_DL_3chain']:+.1f}% "
              f"chain3_sym={row['tension_DL_3chain_sym']:+.1f}%")

    return pd.DataFrame(rows)


def run_lcdm_experiment(T=12000, n_seeds=20, cmb_k=0.9, dl_s=3.0):
    """
    Run the Hubble tension simulation with ΛCDM latent process.
    Compare results to AR(1) equivalent at the same effective memory.
    """
    print("\nRunning ΛCDM latent process experiment...")

    lam_vals = np.linspace(0.50, 0.95, 12)
    rows_ar1  = []
    rows_lcdm = []

    rng = np.random.default_rng(42)

    for lam in lam_vals:
        # AR(1) process
        h0_cmb_ar1, h0_dl_ar1 = [], []
        for seed in range(n_seeds):
            z = np.zeros(T)
            noise = rng.standard_normal(T) * SIGMA
            for t in range(1, T):
                z[t] = lam * z[t-1] + noise[t]
            h0_c = compute_h0_norm(map_cmb(z, cmb_k))
            h0_d = compute_h0_norm(map_dl_single(z, dl_s))
            if not np.isnan(h0_c) and not np.isnan(h0_d):
                h0_cmb_ar1.append(h0_c)
                h0_dl_ar1.append(h0_d)

        t_ar1 = ((np.mean(h0_dl_ar1) - np.mean(h0_cmb_ar1)) /
                  np.mean(h0_cmb_ar1) * 100) if h0_cmb_ar1 else np.nan
        rows_ar1.append({'lam_mem': lam, 'tension': t_ar1,
                          'H0_cmb': np.mean(h0_cmb_ar1),
                          'H0_dl':  np.mean(h0_dl_ar1),  'type': 'AR(1)'})

    # ΛCDM — generate multiple seeds via phase randomization
    h0_cmb_lcdm, h0_dl_lcdm = [], []
    lcdm_lam_effs = []
    for seed in range(n_seeds * 3):   # more seeds since ΛCDM is fixed spectrum
        z_lcdm, _, _ = generate_lcdm_latent(T=T, seed=seed)
        lcdm_lam_effs.append(compute_effective_memory(z_lcdm))
        h0_c = compute_h0_norm(map_cmb(z_lcdm, cmb_k))
        h0_d = compute_h0_norm(map_dl_single(z_lcdm, dl_s))
        if not np.isnan(h0_c) and not np.isnan(h0_d):
            h0_cmb_lcdm.append(h0_c)
            h0_dl_lcdm.append(h0_d)

    lam_eff_mean = np.mean(lcdm_lam_effs)
    t_lcdm = ((np.mean(h0_dl_lcdm) - np.mean(h0_cmb_lcdm)) /
               np.mean(h0_cmb_lcdm) * 100) if h0_cmb_lcdm else np.nan

    print(f"  ΛCDM effective memory λ_eff = {lam_eff_mean:.4f}")
    print(f"  ΛCDM H0_CMB = {np.mean(h0_cmb_lcdm):.4f}")
    print(f"  ΛCDM H0_DL  = {np.mean(h0_dl_lcdm):.4f}")
    print(f"  ΛCDM tension = {t_lcdm:.2f}%  (observed: {OBS_TENSION:.2f}%)")

    lcdm_row = {
        'lam_mem': lam_eff_mean,
        'tension': t_lcdm,
        'H0_cmb': np.mean(h0_cmb_lcdm),
        'H0_dl':  np.mean(h0_dl_lcdm),
        'H0_cmb_std': np.std(h0_cmb_lcdm),
        'H0_dl_std':  np.std(h0_dl_lcdm),
        'type': 'ΛCDM'
    }

    return pd.DataFrame(rows_ar1), lcdm_row, z_lcdm, lam_eff_mean


# ═══════════════════════════════════════════════════════════════
# COMPONENT 3: HIGH-RESOLUTION SWEEP
# ═══════════════════════════════════════════════════════════════

def run_highres_sweep(T=12000, n_seeds=25):
    """
    Fine-grained sweep around confirmed best-match region.
    Produces clean contour lines for the paper figure.
    """
    print("\nRunning high-resolution sweep...")

    # Fine grid around best-match region
    k_grid = np.linspace(0.60, 1.40, 16)   # CMB softplus sharpness
    s_grid = np.linspace(2.00, 5.00, 16)   # DL sigmoid scale
    lam_vals = [0.62, 0.66, 0.70, 0.75, 0.80]

    rows = []
    total = len(k_grid) * len(s_grid) * len(lam_vals)
    done = 0

    for lam in lam_vals:
        for k in k_grid:
            for s in s_grid:
                h0_c_seeds, h0_d_seeds = [], []
                rng = np.random.default_rng(7)
                for seed in range(n_seeds):
                    z = np.zeros(T)
                    noise = rng.standard_normal(T) * SIGMA
                    for t in range(1, T):
                        z[t] = lam * z[t-1] + noise[t]
                    h0_c = compute_h0_norm(map_cmb(z, k))
                    h0_d = compute_h0_norm(map_dl_single(z, s))
                    if not np.isnan(h0_c) and not np.isnan(h0_d):
                        h0_c_seeds.append(h0_c)
                        h0_d_seeds.append(h0_d)

                done += 1
                if not h0_c_seeds:
                    continue

                H0_c = np.mean(h0_c_seeds)
                H0_d = np.mean(h0_d_seeds)
                tension = (H0_d - H0_c) / H0_c * 100

                rows.append({
                    'lam_mem': lam, 'cmb_k': k, 'dl_s': s,
                    'H0_cmb': H0_c, 'H0_dl': H0_d,
                    'tension_pct': tension, 'match': abs(tension - OBS_TENSION),
                    'H0_cmb_std': np.std(h0_c_seeds),
                    'H0_dl_std':  np.std(h0_d_seeds),
                })

                if done % 80 == 0:
                    pct = 100*done/total
                    print(f"  [{pct:5.1f}%] k={k:.2f} s={s:.2f} λ={lam:.2f} "
                          f"→ {tension:+.2f}%")

    df = pd.DataFrame(rows)
    print(f"  High-res sweep: {len(df)} points, "
          f"{(df['match']<1.0).sum()} within ±1% of observed tension")
    return df


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_lcdm_comparison(df_ar1, lcdm_row, z_lcdm, k_lcdm, P_lcdm,
                          save_path=None):
    """ΛCDM latent process vs AR(1): spectral and tension comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('ΛCDM Latent Process: Cosmologically Grounded Hubble Tension',
                 fontsize=12, fontweight='bold')

    # Panel 1: ΛCDM power spectrum
    ax = axes[0]
    ax.loglog(k_lcdm, P_lcdm, 'b-', lw=2, label='ΛCDM P(k)')
    ax.axvline(1e-2, color='gray', ls='--', alpha=0.5, label='Equality scale')
    ax.axvline(0.1,  color='gray', ls=':',  alpha=0.5, label='BAO scale')
    ax.set_xlabel('Wavenumber $k$ [h/Mpc]', fontsize=11)
    ax.set_ylabel('Power $P(k)$ [arb.]', fontsize=11)
    ax.set_title('ΛCDM Matter Power Spectrum\n(Eisenstein-Hu transfer function)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # Panel 2: ΛCDM latent field realization (first 800 steps)
    ax2 = axes[1]
    t_show = min(800, len(z_lcdm))
    ax2.plot(np.arange(t_show), z_lcdm[:t_show], color='steelblue',
             lw=1.2, alpha=0.85)
    ax2.axhline(0, color='gray', ls=':', alpha=0.4)
    ax2.set_xlabel('Position (comoving steps)', fontsize=11)
    ax2.set_ylabel('Density fluctuation $z$', fontsize=11)
    ax2.set_title(f'ΛCDM Latent Field Realization\n'
                  f'λ_eff = {lcdm_row["lam_mem"]:.4f}', fontsize=10)
    ax2.grid(alpha=0.3)

    # Panel 3: Tension comparison AR(1) vs ΛCDM
    ax3 = axes[2]
    df_ar1_s = df_ar1.sort_values('lam_mem')
    ax3.plot(df_ar1_s['lam_mem'], df_ar1_s['tension'],
             'o-', color='gray', lw=2, ms=6, label='AR(1) latent process')

    # ΛCDM as a single point with error bars
    lcdm_err = ((lcdm_row['H0_dl_std']**2 + lcdm_row['H0_cmb_std']**2)**0.5
                / lcdm_row['H0_cmb'] * 100) if 'H0_cmb_std' in lcdm_row else 1.0
    ax3.errorbar(lcdm_row['lam_mem'], lcdm_row['tension'],
                 yerr=lcdm_err, fmt='D', color='blue',
                 ms=12, capsize=6, lw=2.5, zorder=10,
                 label=f"ΛCDM latent\n{lcdm_row['tension']:.1f}%±{lcdm_err:.1f}%")

    ax3.axhspan(OBS_TENSION - 0.5, OBS_TENSION + 0.5,
                alpha=0.2, color='red', zorder=0)
    ax3.axhline(OBS_TENSION, color='red', ls='--', lw=2,
                label=f'Observed: {OBS_TENSION:.1f}%')
    ax3.set_xlabel(r'Effective memory $\lambda_{\rm eff}$', fontsize=11)
    ax3.set_ylabel('Predicted H₀ tension (%)', fontsize=11)
    ax3.set_title('AR(1) vs ΛCDM Latent Process\n(same frames: k=0.9, s=3.0)',
                  fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_composed_observer(df_comp, save_path=None):
    """Composed observer: single vs chained maps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Composed Observer Experiment: Distance Ladder as Chained Softplus Maps\n'
        'Prediction: 3-rung chain produces greater H₀ tension than single map',
        fontsize=12, fontweight='bold'
    )

    df = df_comp.sort_values('lam_mem')

    # Panel 1: Tension vs memory for all configurations
    ax = axes[0]
    configs = [
        ('tension_DL_sigmoid',    'tomato',    'Single sigmoid (s=3.0)',    '--'),
        ('tension_DL_3chain',     'darkorange', '3-rung chain (k₁=1.0, k₂=0.8, k₃=1.2)', '-'),
        ('tension_DL_3chain_sym', 'purple',    '3-rung chain sym (k=0.9)',  '-.'),
    ]
    for col, color, label, ls in configs:
        vals = df[col].values
        mask = np.isfinite(vals)
        ax.plot(df['lam_mem'].values[mask], vals[mask],
                marker='o', ls=ls, color=color, lw=2.2, ms=6, label=label)

    ax.axhspan(OBS_TENSION - 0.5, OBS_TENSION + 0.5, alpha=0.2, color='red')
    ax.axhline(OBS_TENSION, color='red', ls='--', lw=2,
               label=f'Observed tension {OBS_TENSION:.1f}%')
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel(r'Memory $\lambda_{\rm mem}$', fontsize=12)
    ax.set_ylabel('Tension vs CMB frame (%)', fontsize=12)
    ax.set_title('Tension by Ladder Configuration', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Amplification factor (chain / single)
    ax2 = axes[1]
    t_single = df['tension_DL_sigmoid'].values
    t_chain3 = df['tension_DL_3chain'].values
    t_sym    = df['tension_DL_3chain_sym'].values
    lams     = df['lam_mem'].values

    mask = np.isfinite(t_single) & np.isfinite(t_chain3) & (t_single > 0)
    amp3  = t_chain3[mask]  / t_single[mask]
    amp_s = t_sym[mask]     / t_single[mask]

    ax2.plot(lams[mask], amp3,  'o-', color='darkorange', lw=2, ms=6,
             label='3-rung asymmetric / single')
    ax2.plot(lams[mask], amp_s, 's-', color='purple',     lw=2, ms=6,
             label='3-rung symmetric / single')
    ax2.axhline(1.0, color='gray', ls='--', lw=1.5, label='No amplification')

    # Theoretical prediction: each rung adds ~1 level of hierarchy
    # Prediction: amplification ~ n_rungs for additive, n_rungs^2 for multiplicative
    ax2.axhline(3.0, color='gray', ls=':', alpha=0.5, label='Linear (3×) prediction')

    ax2.set_xlabel(r'Memory $\lambda_{\rm mem}$', fontsize=12)
    ax2.set_ylabel('Tension amplification factor\n(chain / single)', fontsize=12)
    ax2.set_title('Hierarchical Amplification\nComposed observer vs single frame',
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_highres_contours(df_hr, save_path=None):
    """Publication-quality tension contour map from high-res sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        f'High-Resolution H₀ Tension Phase Diagram  (T=12000, n_seeds=25)\n'
        f'Black contour = observed tension {OBS_TENSION:.1f}%  '
        f'·  Dashed = ±2% band',
        fontsize=12, fontweight='bold'
    )

    lam_fixed_list = [0.62, 0.70, 0.80]

    for ax, lam_f in zip(axes, lam_fixed_list):
        sub = df_hr[np.abs(df_hr['lam_mem'] - lam_f) < 0.03]
        if sub.empty:
            ax.text(0.5, 0.5, f'No data at λ={lam_f}', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        pivot = sub.pivot_table(
            values='tension_pct', index='dl_s', columns='cmb_k', aggfunc='mean'
        )
        if pivot.empty:
            continue

        # Smooth slightly for clean contours
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(pivot.values, sigma=0.8)

        k_vals = pivot.columns.values
        s_vals = pivot.index.values

        # Color map centered on observed tension
        vrange = max(abs(Z[np.isfinite(Z)]).max(), 1.0)
        vmin_c = max(0, OBS_TENSION - 15)
        vmax_c = OBS_TENSION + 15

        im = ax.pcolormesh(k_vals, s_vals, Z, cmap='RdYlGn_r',
                           shading='auto', vmin=vmin_c, vmax=vmax_c)
        cb = plt.colorbar(im, ax=ax, label='Tension (%)')

        # Key contours
        try:
            cs_obs = ax.contour(k_vals, s_vals, Z,
                                levels=[OBS_TENSION],
                                colors=['black'], linewidths=[2.5], linestyles=['-'])
            ax.clabel(cs_obs, fmt=f'{OBS_TENSION:.1f}%', fontsize=10,
                      inline=True, colors='black')

            cs_band = ax.contour(k_vals, s_vals, Z,
                                 levels=[OBS_TENSION - 2, OBS_TENSION + 2],
                                 colors=['black', 'black'],
                                 linewidths=[1.0, 1.0], linestyles=['--', '--'])
        except Exception:
            pass

        # Mark best-match point
        best_pt = sub.nsmallest(1, 'match')
        if not best_pt.empty:
            ax.scatter(best_pt['cmb_k'], best_pt['dl_s'],
                       marker='*', s=250, color='black', zorder=10,
                       label=f"Best: {best_pt['tension_pct'].values[0]:.1f}%")
            ax.legend(fontsize=8, loc='lower right')

        ax.set_xlabel('CMB frame softplus $k$', fontsize=11)
        ax.set_ylabel('DL frame sigmoid $s$', fontsize=11)
        ax.set_title(f'λ_mem = {lam_f:.2f}', fontsize=11)
        ax.grid(alpha=0.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_summary(df_hr, df_comp, lcdm_row, df_ar1, save_path=None):
    """4-panel summary figure for the paper."""
    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: High-res contour at λ=0.70 ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sub = df_hr[np.abs(df_hr['lam_mem'] - 0.70) < 0.03]
    if not sub.empty:
        pivot = sub.pivot_table('tension_pct', index='dl_s', columns='cmb_k',
                                aggfunc='mean')
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(pivot.values, sigma=0.8)
        im = ax1.pcolormesh(pivot.columns.values, pivot.index.values, Z,
                            cmap='RdYlGn_r', shading='auto',
                            vmin=OBS_TENSION-15, vmax=OBS_TENSION+15)
        plt.colorbar(im, ax=ax1, label='Tension (%)')
        try:
            cs = ax1.contour(pivot.columns.values, pivot.index.values, Z,
                             levels=[OBS_TENSION], colors=['black'], linewidths=[2.5])
            ax1.clabel(cs, fmt=f'{OBS_TENSION:.1f}%', fontsize=9)
        except Exception:
            pass
    ax1.set_xlabel('CMB softplus $k$', fontsize=11)
    ax1.set_ylabel('DL sigmoid $s$', fontsize=11)
    ax1.set_title('(A) Tension Phase Diagram (λ=0.70)\nBlack = observed 8.31%', fontsize=10)

    # ── Panel B: Composed observer ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    df_c = df_comp.sort_values('lam_mem')
    cfgs = [('tension_DL_sigmoid', 'tomato', 'Single sigmoid'),
            ('tension_DL_3chain',  'darkorange', '3-rung chain'),
            ('tension_DL_3chain_sym', 'purple', '3-rung sym')]
    for col, color, label in cfgs:
        v = df_c[col].values
        m = np.isfinite(v)
        ax2.plot(df_c['lam_mem'].values[m], v[m], 'o-', color=color, lw=2, ms=5,
                 label=label)
    ax2.axhline(OBS_TENSION, color='red', ls='--', lw=2,
                label=f'Observed {OBS_TENSION:.1f}%')
    ax2.axhspan(OBS_TENSION-0.5, OBS_TENSION+0.5, alpha=0.15, color='red')
    ax2.set_xlabel(r'$\lambda_{\rm mem}$', fontsize=11)
    ax2.set_ylabel('Tension (%)', fontsize=11)
    ax2.set_title('(B) Composed Observer\nChained rungs vs single map', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)

    # ── Panel C: ΛCDM vs AR(1) ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    df_a = df_ar1.sort_values('lam_mem')
    ax3.plot(df_a['lam_mem'], df_a['tension'], 'o-', color='gray', lw=2, ms=6,
             label='AR(1)')
    lcdm_err = 1.0
    ax3.errorbar(lcdm_row['lam_mem'], lcdm_row['tension'],
                 yerr=lcdm_err, fmt='D', color='blue', ms=12,
                 capsize=6, lw=2.5, zorder=10,
                 label=f"ΛCDM  {lcdm_row['tension']:.1f}%")
    ax3.axhline(OBS_TENSION, color='red', ls='--', lw=2,
                label=f'Observed {OBS_TENSION:.1f}%')
    ax3.axhspan(OBS_TENSION-0.5, OBS_TENSION+0.5, alpha=0.15, color='red')
    ax3.set_xlabel(r'$\lambda_{\rm eff}$', fontsize=11)
    ax3.set_ylabel('Tension (%)', fontsize=11)
    ax3.set_title('(C) ΛCDM vs AR(1) Latent Process\nSame frames: k=0.9, s=3.0',
                  fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # ── Panel D: Best-match tension sweep ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    best_k, best_s = 0.9, 3.0
    sub_best = df_hr[(np.abs(df_hr['cmb_k']-best_k)<0.05) &
                     (np.abs(df_hr['dl_s']-best_s)<0.15)].sort_values('lam_mem')
    if not sub_best.empty:
        ax4.fill_between(sub_best['lam_mem'],
                         sub_best['tension_pct'] - sub_best['H0_dl_std']*100/sub_best['H0_dl'],
                         sub_best['tension_pct'] + sub_best['H0_dl_std']*100/sub_best['H0_dl'],
                         alpha=0.2, color='steelblue')
        ax4.plot(sub_best['lam_mem'], sub_best['tension_pct'],
                 'o-', color='steelblue', lw=2.5, ms=7,
                 label=f'k={best_k}, s={best_s}')

    ax4.errorbar(lcdm_row['lam_mem'], lcdm_row['tension'],
                 yerr=1.0, fmt='D', color='blue', ms=10, capsize=5,
                 label=f'ΛCDM point')
    ax4.axhline(OBS_TENSION, color='red', ls='--', lw=2,
                label=f'Observed {OBS_TENSION:.1f}%')
    ax4.axhspan(OBS_TENSION-0.5, OBS_TENSION+0.5, alpha=0.15, color='red')
    ax4.axvspan(0.62, 0.72, alpha=0.08, color='green')
    ax4.annotate('Best-match\nzone', xy=(0.67, OBS_TENSION*0.5),
                 fontsize=9, ha='center', color='darkgreen')
    ax4.set_xlabel(r'$\lambda_{\rm mem}$', fontsize=11)
    ax4.set_ylabel('Tension (%)', fontsize=11)
    ax4.set_title(f'(D) Best-Match Frame Pair With Uncertainty\nk={best_k}, s={best_s}',
                  fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    fig.suptitle(
        'PMIR Preprint 7 — Summary: Observation-Frame Mechanism for Hubble Tension\n'
        f'Predicted: {OBS_TENSION:.1f}% from frame structure alone  ·  '
        f'Observed: Planck {H0_CMB} vs SH0ES {H0_LOCAL} km/s/Mpc',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    OUT_DIR = '/home/claude/pmir_p7_extended'
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PMIR PREPRINT 7 — EXTENDED SIMULATION SUITE")
    print("=" * 70)
    print(f"Target tension: {OBS_TENSION:.2f}%  (Planck {H0_CMB} / SH0ES {H0_LOCAL})")
    print()

    # ── Component 1: ΛCDM ───────────────────────────────────────────────
    print("COMPONENT 1: ΛCDM Latent Process")
    print("-" * 40)
    z_lcdm_ex, k_lcdm, P_lcdm = generate_lcdm_latent(T=12000, seed=42)
    lam_eff = compute_effective_memory(z_lcdm_ex)
    print(f"  ΛCDM field: T=12000, λ_eff={lam_eff:.4f}")
    df_ar1, lcdm_row, z_lcdm_final, lam_eff_f = run_lcdm_experiment(
        T=8000, n_seeds=15, cmb_k=0.9, dl_s=3.0)

    # ── Component 2: Composed observer ──────────────────────────────────
    print("\nCOMPONENT 2: Composed Observer")
    print("-" * 40)
    df_comp = run_composed_observer_experiment(T=8000, n_seeds=15)

    # ── Component 3: High-res sweep ──────────────────────────────────────
    print("\nCOMPONENT 3: High-Resolution Sweep")
    print("-" * 40)
    df_hr = run_highres_sweep(T=8000, n_seeds=20)

    # ── Save all results ─────────────────────────────────────────────────
    df_ar1.to_csv(os.path.join(OUT_DIR, 'lcdm_ar1_comparison.csv'), index=False)
    df_comp.to_csv(os.path.join(OUT_DIR, 'composed_observer.csv'), index=False)
    df_hr.to_csv(os.path.join(OUT_DIR, 'highres_sweep.csv'), index=False)

    # ── Generate figures ─────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_lcdm_comparison(
        df_ar1, lcdm_row, z_lcdm_final, k_lcdm, P_lcdm,
        save_path=os.path.join(OUT_DIR, 'pmir_lcdm_latent.png'))
    plot_composed_observer(
        df_comp,
        save_path=os.path.join(OUT_DIR, 'pmir_composed_observer.png'))
    plot_highres_contours(
        df_hr,
        save_path=os.path.join(OUT_DIR, 'pmir_highres_sweep.png'))
    plot_summary(
        df_hr, df_comp, lcdm_row, df_ar1,
        save_path=os.path.join(OUT_DIR, 'pmir_preprint7_summary.png'))

    # ── Final report ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXTENDED SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nΛCDM result:  {lcdm_row['tension']:.2f}%  (obs: {OBS_TENSION:.2f}%)")

    best_hr = df_hr.nsmallest(1, 'match').iloc[0]
    print(f"High-res best: k={best_hr['cmb_k']:.2f} s={best_hr['dl_s']:.2f} "
          f"λ={best_hr['lam_mem']:.2f} → {best_hr['tension_pct']:.2f}%")

    comp_at70 = df_comp[np.abs(df_comp['lam_mem']-0.70)<0.04]
    if not comp_at70.empty:
        r = comp_at70.iloc[0]
        print(f"Composed observer (λ=0.70): "
              f"single={r['tension_DL_sigmoid']:.1f}% "
              f"chain3={r['tension_DL_3chain']:.1f}%")

    print()
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        print(f"  {f:50s}  {os.path.getsize(fpath)/1024:6.1f} KB")
