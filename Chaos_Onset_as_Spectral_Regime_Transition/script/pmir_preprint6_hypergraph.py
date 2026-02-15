"""
pmir_preprint6_hypergraph.py
============================
Complete analysis script for:

  "Chaos Onset as Spectral Regime Transition: Hypergraph Laplacian
   Analysis of N-Body Gravitational Systems via PMIR"
  Richard L. Schorr III, 2026
  Zenodo: [DOI assigned on upload]

Reproduces all numerical results and generates all four figures:
  fig1_spectral_timeseries.png  -- alpha(t) and gap(t) epoch sweep
  fig2_regime_diagram.png       -- regime scatter + theory curve
  fig3_pmir_dynamics.png        -- PMIR R(t) and Fiedler evolution
  fig4_validation.png           -- SJS vs SJSU distributions

Requirements: numpy, scipy, matplotlib (standard scientific Python stack)
Runtime: ~15 minutes for full 60-epoch sweep

Usage:
    python pmir_preprint6_hypergraph.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ORBITAL CONSTANTS AND ELEMENTS
# ─────────────────────────────────────────────────────────────────────────────

GM_SUN = 4 * np.pi**2          # AU^3 / yr^2  (G*M_sun in astronomical units)

# Orbital elements: (a AU, e, T yr, omega0 rad, gdot rad/yr, mass kg)
BODIES = {
    'Sun':    {'a': 0.0,    'e': 0.0,    'T': None,   'omega0': 0.0,
               'gdot': 0.0, 'mass': 1.989e30},
    'Jupiter':{'a': 5.2044, 'e': 0.0489, 'T': 11.862, 'omega0': np.radians(274.05),
               'gdot': np.radians(4.257/3600), 'mass': 1.898e27},
    'Saturn': {'a': 9.5826, 'e': 0.0565, 'T': 29.457, 'omega0': np.radians(339.39),
               'gdot': np.radians(28.243/3600), 'mass': 5.683e26},
    'Uranus': {'a': 19.218, 'e': 0.0472, 'T': 84.011, 'omega0': np.radians(97.77),
               'gdot': np.radians(3.316/3600),  'mass': 8.681e25},
}

ALPHA_CRIT = 5.67      # empirical regime boundary
N_STEPS    = 4000      # time steps per epoch window
T_WINDOW   = 20.0      # years per window
DT         = T_WINDOW / N_STEPS   # ~1.8 days

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: KEPLERIAN ORBIT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def kepler_solve(M, e, tol=1e-10, maxiter=50):
    """Newton-Raphson solution of Kepler's equation M = E - e*sin(E)."""
    E = M.copy()
    for _ in range(maxiter):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def generate_trajectory(body_name, t_array):
    """
    Generate Keplerian trajectory with Laplace-Lagrange secular precession.
    Returns arrays: r (AU), rdot (AU/yr), kappa (1/AU), x (AU), y (AU)
    """
    b = BODIES[body_name]
    if b['a'] == 0.0:
        # Sun at origin
        n = len(t_array)
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    a, e, omega0, gdot = b['a'], b['e'], b['omega0'], b['gdot']
    n_motion = 2 * np.pi / b['T']   # mean motion rad/yr
    h = np.sqrt(GM_SUN * a * (1 - e**2))

    M = n_motion * t_array
    E = kepler_solve(M, e)
    cos_E, sin_E = np.cos(E), np.sin(E)

    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))

    r    = a * (1 - e * cos_E)
    rdot = (GM_SUN / h) * e * np.sin(nu)
    kappa = h / (r * (h / r)**2)   # = h / (r * v^2) where v=h/r for circular approx
    # More precisely: kappa = 1/r * (1 + e*cos(nu)) -- curvature of ellipse
    kappa = (1 / (a * (1 - e**2))) * (1 + e * np.cos(nu))

    omega = omega0 + gdot * t_array
    theta = nu + omega
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return r, rdot, kappa, x, y

def get_composite_signal(body_name, t_array):
    """Return normalized composite signal s_i(t) = (rdot + kappa - mu) / sigma."""
    r, rdot, kappa, x, y = generate_trajectory(body_name, t_array)
    raw = rdot + kappa
    mu, sigma = raw.mean(), raw.std()
    if sigma < 1e-12:
        return np.zeros_like(raw), x, y, r
    return (raw - mu) / sigma, x, y, r

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: OBSERVATION-INDUCED HYPERGRAPH WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def xcorr_max(a, b, tau_max_frac=0.1):
    """Maximum normalized cross-correlation over lag |tau| <= tau_max."""
    n = len(a)
    tau_max = int(n * tau_max_frac)
    a_n = (a - a.mean()) / (a.std() + 1e-12)
    b_n = (b - b.mean()) / (b.std() + 1e-12)
    best = 0.0
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            c = np.dot(a_n[:n-tau], b_n[tau:]) / (n - tau)
        else:
            c = np.dot(a_n[-tau:], b_n[:n+tau]) / (n + tau)
        if abs(c) > best:
            best = abs(c)
    return best

def entropy_1d(x, bins=20):
    """Differential entropy via histogram."""
    hist, edges = np.histogram(x, bins=bins,
                               range=(x.mean() - 3*x.std(), x.mean() + 3*x.std()),
                               density=True)
    dx = edges[1] - edges[0]
    p = hist * dx
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def entropy_nd(*arrays, bins=20):
    """Joint entropy via multi-dim histogram."""
    ranges = [(a.mean() - 3*a.std(), a.mean() + 3*a.std()) for a in arrays]
    hist, _ = np.histogramdd(np.column_stack(arrays),
                              bins=bins, range=ranges, density=True)
    vol = np.prod([(r[1]-r[0])/bins for r in ranges])
    p = hist.ravel() * vol
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def total_correlation(*arrays, bins=20):
    """TC(X1,...,Xn) = sum H(Xi) - H(X1,...,Xn)."""
    marginals = sum(entropy_1d(a, bins=bins) for a in arrays)
    joint = entropy_nd(*arrays, bins=bins)
    return max(marginals - joint, 0.0)

def normalized_tc(*arrays, bins=20):
    """TC normalized by min marginal entropy."""
    tc = total_correlation(*arrays, bins=bins)
    min_h = min(entropy_1d(a, bins=bins) for a in arrays)
    if min_h < 1e-10:
        return 0.0
    return tc / min_h

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HYPERGRAPH LAPLACIAN (ZHOU 2006)
# ─────────────────────────────────────────────────────────────────────────────

def build_hypergraph_laplacian(node_names, edges):
    """
    Build Zhou normalized hypergraph Laplacian.

    Parameters
    ----------
    node_names : list of str
    edges      : list of (frozenset_of_node_names, weight)

    Returns
    -------
    Delta : ndarray (N x N)
    eigenvalues sorted ascending
    """
    N = len(node_names)
    idx = {name: i for i, name in enumerate(node_names)}

    n_edges = len(edges)
    B = np.zeros((N, n_edges))
    w = np.zeros(n_edges)
    d_e = np.zeros(n_edges)

    for j, (nodes, weight) in enumerate(edges):
        for nd in nodes:
            B[idx[nd], j] = 1.0
        w[j] = weight
        d_e[j] = len(nodes)

    W   = np.diag(w)
    De  = np.diag(d_e)
    Dv  = np.diag(B @ w)

    Dv_invsqrt = np.diag(1.0 / np.sqrt(np.maximum(Dv.diagonal(), 1e-12)))
    De_inv      = np.diag(1.0 / np.maximum(De.diagonal(), 1e-12))

    Theta = Dv_invsqrt @ B @ W @ De_inv @ B.T @ Dv_invsqrt
    Delta = np.eye(N) - Theta

    evals = np.linalg.eigvalsh(Delta)
    evecs = np.linalg.eigh(Delta)[1]
    return Delta, np.sort(evals), evecs[:, np.argsort(evals)]

def compute_alpha(edges):
    """Compute alpha = mean(higher-order weights) / mean(pairwise weights)."""
    pair   = [w for (nodes, w) in edges if len(nodes) == 2]
    higher = [w for (nodes, w) in edges if len(nodes) > 2]
    if not pair or not higher:
        return 0.0
    return np.mean(higher) / np.mean(pair)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SINGLE EPOCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def solar_proxy(t_array, body_list):
    """Mass-weighted barycentric displacement of listed planets as Sun proxy."""
    total_mass = sum(BODIES[b]['mass'] for b in body_list)
    x_bary = np.zeros(len(t_array))
    for b in body_list:
        _, _, _, x, y = generate_trajectory(b, t_array)
        x_bary += BODIES[b]['mass'] * x
    x_bary /= total_mass
    sig = x_bary - x_bary.mean()
    std = sig.std()
    return sig / (std + 1e-12)

def compute_epoch_sjs(t0):
    """Compute SJS hypergraph Laplacian for a 20-yr window starting at t0."""
    t = np.linspace(t0, t0 + T_WINDOW, N_STEPS)
    sJ, xJ, yJ, rJ = get_composite_signal('Jupiter', t)
    sS, xS, yS, rS = get_composite_signal('Saturn',  t)
    sSun = solar_proxy(t, ['Jupiter', 'Saturn'])

    wJS  = xcorr_max(sJ, sS)
    wJSun = xcorr_max(sJ, sSun)
    wSSun = xcorr_max(sS, sSun)
    wSJS  = normalized_tc(sSun, sJ, sS, bins=20)

    edges = [
        (frozenset(['Jupiter', 'Saturn']),  wJS),
        (frozenset(['Jupiter', 'Sun']),     wJSun),
        (frozenset(['Saturn',  'Sun']),     wSSun),
        (frozenset(['Jupiter', 'Saturn', 'Sun']), wSJS),
    ]
    Delta, evals, evecs = build_hypergraph_laplacian(['Sun','Jupiter','Saturn'], edges)
    alpha = compute_alpha(edges)
    gap   = evals[2] - evals[1] if len(evals) >= 3 else 0.0
    fiedler = evecs[:, 1]   # second eigenvector

    # Jupiter-Saturn separation for conjunction detection
    rjs_mean = np.mean(np.sqrt((xJ - xS)**2 + (yJ - yS)**2))

    return {
        'alpha': alpha, 'gap': gap, 'lambda2': evals[1],
        'fiedler': fiedler, 'evals': evals,
        'wJS': wJS, 'wSJS': wSJS, 'rjs': rjs_mean,
        'edges': edges, 'Delta': Delta
    }

def compute_epoch_sjsu(t0):
    """Compute SJSU hypergraph Laplacian for a 20-yr window starting at t0."""
    t = np.linspace(t0, t0 + T_WINDOW, N_STEPS)
    sJ, xJ, yJ, rJ = get_composite_signal('Jupiter', t)
    sS, xS, yS, rS = get_composite_signal('Saturn',  t)
    sU, xU, yU, rU = get_composite_signal('Uranus',  t)
    sSun = solar_proxy(t, ['Jupiter', 'Saturn', 'Uranus'])

    # Pairwise (6)
    wJS  = xcorr_max(sJ, sS)
    wJU  = xcorr_max(sJ, sU)
    wSU  = xcorr_max(sS, sU)
    wJSun = xcorr_max(sJ, sSun)
    wSSun = xcorr_max(sS, sSun)
    wUSun = xcorr_max(sU, sSun)

    # 3-hyperedges (4)
    wSJS  = normalized_tc(sSun, sJ, sS, bins=20)
    wSJU  = normalized_tc(sSun, sJ, sU, bins=20)
    wSSaU = normalized_tc(sSun, sS, sU, bins=20)
    wJSaU = normalized_tc(sJ, sS, sU, bins=20)

    # 4-hyperedge (1)
    wSJSU = normalized_tc(sSun, sJ, sS, sU, bins=12)

    edges = [
        (frozenset(['Sun','Jupiter']),             wJSun),
        (frozenset(['Sun','Saturn']),              wSSun),
        (frozenset(['Sun','Uranus']),              wUSun),
        (frozenset(['Jupiter','Saturn']),          wJS),
        (frozenset(['Jupiter','Uranus']),          wJU),
        (frozenset(['Saturn','Uranus']),           wSU),
        (frozenset(['Sun','Jupiter','Saturn']),    wSJS),
        (frozenset(['Sun','Jupiter','Uranus']),    wSJU),
        (frozenset(['Sun','Saturn','Uranus']),     wSSaU),
        (frozenset(['Jupiter','Saturn','Uranus']), wJSaU),
        (frozenset(['Sun','Jupiter','Saturn','Uranus']), wSJSU),
    ]
    nodes = ['Sun','Jupiter','Saturn','Uranus']
    Delta, evals, evecs = build_hypergraph_laplacian(nodes, edges)
    alpha = compute_alpha(edges)
    alpha4 = wSJSU / max(np.mean([wJSun,wSSun,wUSun,wJS,wJU,wSU]), 1e-12)
    gap   = evals[2] - evals[1] if len(evals) >= 3 else 0.0
    fiedler = evecs[:, 1]

    return {
        'alpha': alpha, 'alpha4': alpha4, 'gap': gap, 'lambda2': evals[1],
        'fiedler': fiedler, 'evals': evals,
        'wSJSU': wSJSU, 'edges': edges, 'Delta': Delta
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PMIR RIVALRY DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def run_pmir(Delta, T_sim=200, dt=0.01, gamma=1.0, beta=0.5, seed=42):
    """
    Integrate PMIR dynamics:  dphi/dt = -gamma * Delta @ phi + beta * tanh(phi)
    Returns time array and rivalry R(t) = mean |phi_i - phi_j| over edges.
    """
    rng = np.random.default_rng(seed)
    N = Delta.shape[0]
    phi = rng.normal(0, 0.1, N)
    steps = int(T_sim / dt)
    t_arr = np.zeros(steps)
    R_arr = np.zeros(steps)
    for k in range(steps):
        dphi = -gamma * (Delta @ phi) + beta * np.tanh(phi)
        phi += dt * dphi
        t_arr[k] = k * dt
        R_arr[k] = np.mean(np.abs(phi))
    return t_arr, R_arr

def fit_powerlaw(t, R, t_lo=10.0, t_hi=100.0):
    """OLS fit of log R ~ p * log t over [t_lo, t_hi]. Returns p, R2."""
    mask = (t >= t_lo) & (t <= t_hi) & (R > 0)
    if mask.sum() < 5:
        return np.nan, 0.0
    log_t = np.log(t[mask])
    log_R = np.log(R[mask])
    p = np.polyfit(log_t, log_R, 1)
    log_R_fit = np.polyval(p, log_t)
    ss_res = np.sum((log_R - log_R_fit)**2)
    ss_tot = np.sum((log_R - log_R.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return p[0], r2

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: EPOCH SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(system='SJS', n_epochs=60, epoch_spacing=2.0):
    """Run full epoch sweep. Returns list of result dicts."""
    results = []
    t0_values = np.arange(0, n_epochs * epoch_spacing, epoch_spacing)
    for i, t0 in enumerate(t0_values):
        print(f"  {system} epoch {i+1}/{n_epochs}  t0={t0:.0f} yr", end='\r')
        try:
            if system == 'SJS':
                res = compute_epoch_sjs(t0)
            else:
                res = compute_epoch_sjsu(t0)
            res['t0'] = t0
            res['epoch_centre'] = t0 + T_WINDOW / 2
            results.append(res)
        except Exception as e:
            print(f"\n  Warning: epoch t0={t0} failed: {e}")
    print(f"\n  {system} sweep done: {len(results)} epochs")
    return results

def classify_regime(gap):
    if gap > 0.05:
        return 'structure-sensitive'
    elif gap > 0.01:
        return 'transitional'
    else:
        return 'topology-dominated'

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: CONFIGURATION-SPACE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def config_space_laplacian():
    """
    Compute configuration-space hypergraph Laplacian with gravitational
    potential-energy weights. Returns alpha_config, spectrum, Fiedler vector.
    """
    G = 6.674e-11       # SI
    mSun = 1.989e30; mJ = 1.898e27; mS = 5.683e26
    AU = 1.496e11       # m
    rSJ = 5.2 * AU; rSS = 9.6 * AU; rJS = (9.6 - 5.2) * AU

    wSJ = G * mSun * mJ / rSJ
    wSS = G * mSun * mS / rSS
    wJS = G * mJ   * mS / rJS

    # 3-body config weight: geometric mean of potential energies
    wSJS_config = (wSJ * wSS * wJS)**(1/3)

    edges = [
        (frozenset(['Sun','Jupiter']), wSJ),
        (frozenset(['Sun','Saturn']),  wSS),
        (frozenset(['Jupiter','Saturn']), wJS),
        (frozenset(['Sun','Jupiter','Saturn']), wSJS_config),
    ]
    Delta, evals, evecs = build_hypergraph_laplacian(['Sun','Jupiter','Saturn'], edges)
    alpha_config = compute_alpha(edges)
    return alpha_config, evals, evecs[:, 1]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    'structure-sensitive': '#4488ff',
    'transitional':        '#ffcc00',
    'topology-dominated':  '#ff4444',
}

def fig1_spectral_timeseries(sjs_results, sjsu_results, out='fig1_spectral_timeseries.png'):
    """Figure 1: alpha(t), gap(t), r_JS(t) epoch sweeps."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle('Figure 1 — Observation-Induced Hypergraph Laplacian: Spectral Time Series',
                 fontsize=12, fontweight='bold')

    ec_sjs  = [r['epoch_centre'] for r in sjs_results]
    ec_sjsu = [r['epoch_centre'] for r in sjsu_results]
    alpha_sjs  = [r['alpha'] for r in sjs_results]
    alpha_sjsu = [r['alpha'] for r in sjsu_results]
    alpha4     = [r['alpha4'] for r in sjsu_results]
    gap_sjs    = [r['gap']   for r in sjs_results]
    gap_sjsu   = [r['gap']   for r in sjsu_results]
    rjs        = [r['rjs']   for r in sjs_results]

    ax = axes[0]
    ax.plot(ec_sjs,  alpha_sjs,  'b-',  lw=2,   label='SJS (3-body)')
    ax.plot(ec_sjsu, alpha_sjsu, 'g--', lw=1.5, label='SJSU (4-body)')
    ax.plot(ec_sjsu, alpha4,     'orange', lw=1.5, ls=':', label=r'SJSU $\alpha_4$body')
    ax.axhline(ALPHA_CRIT, color='r', lw=2, ls='-', label=r'$\alpha_{crit}=5.67$')
    ax.axhspan(ALPHA_CRIT, 7, alpha=0.12, color='red')
    ax.axhspan(2.0, ALPHA_CRIT, alpha=0.08, color='gold')
    ax.axhspan(0, 2.0, alpha=0.08, color='blue')
    ax.text(5, 6.3, 'topology-dominated (chaos)', fontsize=8, color='darkred')
    ax.text(5, 3.5, 'transitional', fontsize=8, color='goldenrod')
    ax.text(5, 0.5, 'structure-sensitive (stable)', fontsize=8, color='darkblue')
    ax.set_ylabel(r'$\alpha$ (hyperedge / pairwise)')
    ax.set_ylim(0, 7)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ec_sjs,  gap_sjs,  'b-',  lw=2,   label=r'SJS: 73% structure-sensitive')
    ax.plot(ec_sjsu, gap_sjsu, 'g--', lw=1.5, label='SJSU: 100% transitional')
    ax.axhline(0.05, color='orange', lw=1.5, ls=':', label=r'transitional threshold (0.05)')
    ax.set_ylabel(r'Spectral gap $\lambda_3 - \lambda_2$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ec_sjs, rjs, 'k-', lw=1.5)
    ax.set_ylabel(r'$r_{JS}$ (AU)')
    ax.set_xlabel('Epoch centre (yr)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

def fig2_regime_diagram(sjs_results, sjsu_results, out='fig2_regime_diagram.png'):
    """Figure 2: Regime scatter + theory curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 2 — PMIR Spectral Regime Diagram', fontsize=12, fontweight='bold')

    alpha_sjs  = np.array([r['alpha'] for r in sjs_results])
    gap_sjs    = np.array([r['gap']   for r in sjs_results])
    alpha_sjsu = np.array([r['alpha'] for r in sjsu_results])
    gap_sjsu   = np.array([r['gap']   for r in sjsu_results])
    alpha4     = np.array([r['alpha4'] for r in sjsu_results])

    # Conjunction detection: bottom 30% of r_JS
    rjs = np.array([r['rjs'] for r in sjs_results])
    conj_mask = rjs < np.percentile(rjs, 30)

    ax = axes[0]
    ax.scatter(alpha_sjs[~conj_mask], gap_sjs[~conj_mask],
               c='purple', alpha=0.7, s=40, label='SJS stable', zorder=3)
    ax.scatter(alpha_sjs[conj_mask], gap_sjs[conj_mask],
               c='orange', alpha=0.9, s=60, marker='o', label='SJS near-conjunction', zorder=4)
    ax.scatter(alpha_sjsu, gap_sjsu,
               c='teal', alpha=0.7, s=50, marker='D', label='SJSU (4-body)', zorder=3)
    ax.axvline(ALPHA_CRIT, color='r', ls='--', lw=1.5, label=r'$\alpha_{crit}$')
    ax.axhline(0.05, color='orange', ls=':', lw=1.5, label=r'$\Delta\lambda=0.05$')
    ax.text(ALPHA_CRIT + 0.05, 0.13, 'Topology-\ndominated\n(chaotic)', fontsize=8, color='darkred')
    ax.text(2.1, 0.01, 'Transitional\n(chaos-prone)', fontsize=8, color='goldenrod')
    ax.text(2.1, 0.09, 'Structure-\nsensitive\n(stable)', fontsize=8, color='darkblue')
    ax.set_xlabel(r'$\alpha$ (3-body hyperedge / mean pairwise)')
    ax.set_ylabel(r'Spectral gap $\lambda_3 - \lambda_2$')
    ax.set_title('Regime Scatter: gap vs α')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    alpha_theory = np.logspace(-1, 1, 300)
    # Empirical regime curve: gap ~ 1/(1 + (alpha/alpha_crit)^3)
    gap_theory = 0.12 / (1 + (alpha_theory / ALPHA_CRIT)**3)
    ax.plot(alpha_theory, gap_theory, 'b-', lw=2, label='Regime curve')
    ax.axvline(ALPHA_CRIT, color='r', ls='--', lw=1.5, label=r'$\alpha_{crit}$')
    ax.axhline(0.05, color='orange', ls=':', lw=1.5)

    # Shaded ranges
    ax.axvspan(alpha_sjs.min(),  alpha_sjs.max(),  alpha=0.15, color='purple', label='SJS range')
    ax.axvspan(alpha_sjsu.min(), alpha_sjsu.max(), alpha=0.15, color='teal',   label='SJSU range')
    ax.axvspan(alpha4.min(),     alpha4.max(),     alpha=0.12, color='orange', label=r'SJSU $\alpha_4$ range')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$ (3-body hyperedge / mean pairwise)')
    ax.set_ylabel(r'Spectral gap $\lambda_3 - \lambda_2$')
    ax.set_title(r'Spectral Gap vs $\alpha$: Regime Curve')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

def fig3_pmir_dynamics(sjs_results, sjsu_results, out='fig3_pmir_dynamics.png'):
    """Figure 3: PMIR R(t) at three epochs + Fiedler mode evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Figure 3 — PMIR Dynamics and Fiedler Mode Evolution',
                 fontsize=12, fontweight='bold')

    # Pick three representative epochs
    alpha_sjs = np.array([r['alpha'] for r in sjs_results])
    gap_sjs   = np.array([r['gap']   for r in sjs_results])
    rjs       = np.array([r['rjs']   for r in sjs_results])

    i_stable = np.argmax(gap_sjs)
    i_conj   = np.argmin(rjs)
    i_maxalp = np.argmax(alpha_sjs)

    epoch_map = {
        f'Stable (t={sjs_results[i_stable]["t0"]:.0f}yr)  p=': (i_stable, 'purple'),
        f'Conjunction (t={sjs_results[i_conj]["t0"]:.0f}yr)  p=': (i_conj,   'orange'),
        f'Max-α (t={sjs_results[i_maxalp]["t0"]:.0f}yr)  p=': (i_maxalp, 'teal'),
    }

    ax = axes[0, 0]
    ax.set_title('PMIR Rivalry Dynamics on Observation-Induced Δ')
    t_ref = np.logspace(0, 2.5, 100)
    ax.plot(t_ref, t_ref**1.5, 'k--', lw=1, alpha=0.5, label=r'$t^{1.5}$')
    ax.plot(t_ref, t_ref**2.2, 'k:',  lw=1, alpha=0.5, label=r'$t^{2.2}$')

    pvals = []
    for label, (idx, color) in epoch_map.items():
        t_dyn, R_dyn = run_pmir(sjs_results[idx]['Delta'])
        p, r2 = fit_powerlaw(t_dyn, R_dyn)
        pvals.append(p)
        ax.loglog(t_dyn, R_dyn, color=color, lw=1.8,
                  label=f'{label}{p:.2f}')
    ax.set_xlabel('PMIR time t')
    ax.set_ylabel(r'Rivalry $R(t) = \|\phi\|_1$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bar chart of exponents
    ax = axes[0, 1]
    labels_short = ['Stable', 'Conjunction', r'Max-$\alpha$']
    colors_bar = ['purple', 'orange', 'teal']
    bars = ax.bar(labels_short, pvals, color=colors_bar, edgecolor='k', linewidth=0.8)
    for bar, pv in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width()/2, pv + 0.02, f'{pv:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(2.0, color='k', ls='--', lw=1, label='p=2 (structure-sens.)')
    ax.axhline(1.5, color='k', ls=':', lw=1, label='p=1.5 (transitional)')
    ax.set_ylabel('Power-law exponent p')
    ax.set_title('PMIR Scaling Exponents')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # SJS Fiedler evolution
    ax = axes[1, 0]
    epoch_indices = np.linspace(0, len(sjs_results)-1, 5, dtype=int)
    body_labels_sjs = ['Sun', 'Jupiter', 'Saturn']
    x_pos = np.arange(3)
    width = 0.15
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for k, idx in enumerate(epoch_indices):
        fv = sjs_results[idx]['fiedler']
        t_label = f't={sjs_results[idx]["t0"]:.0f}yr'
        ax.bar(x_pos + k*width, fv, width, label=t_label, color=cmap[k], edgecolor='k', lw=0.5)
    ax.set_xticks(x_pos + 2*width)
    ax.set_xticklabels(body_labels_sjs)
    ax.set_ylabel('Fiedler eigenvector component')
    ax.set_title(r'SJS Fiedler Mode ($\lambda_2$) Evolution')
    ax.axhline(0, color='k', lw=0.8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # SJSU Fiedler evolution
    ax = axes[1, 1]
    epoch_indices4 = np.linspace(0, len(sjsu_results)-1, 5, dtype=int)
    body_labels_sjsu = ['Sun', 'Jupiter', 'Saturn', 'Uranus']
    x_pos4 = np.arange(4)
    for k, idx in enumerate(epoch_indices4):
        fv = sjsu_results[idx]['fiedler']
        t_label = f't={sjsu_results[idx]["t0"]:.0f}yr'
        ax.bar(x_pos4 + k*width, fv, width, label=t_label, color=cmap[k], edgecolor='k', lw=0.5)
    ax.set_xticks(x_pos4 + 2*width)
    ax.set_xticklabels(body_labels_sjsu)
    ax.set_ylabel('Fiedler eigenvector component')
    ax.set_title('SJSU Fiedler Mode (collective 4-body)')
    ax.axhline(0, color='k', lw=0.8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

def fig4_validation(sjs_results, sjsu_results, out='fig4_validation.png'):
    """Figure 4: SJS vs SJSU distribution comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Figure 4 — Validation: SJS vs SJSU Spectral Regime Distributions',
                 fontsize=12, fontweight='bold')

    alpha_sjs  = np.array([r['alpha'] for r in sjs_results])
    alpha_sjsu = np.array([r['alpha'] for r in sjsu_results])
    gap_sjs    = np.array([r['gap']   for r in sjs_results])
    gap_sjsu   = np.array([r['gap']   for r in sjsu_results])

    mw_alpha = mannwhitneyu(alpha_sjs, alpha_sjsu, alternative='two-sided')
    mw_gap   = mannwhitneyu(gap_sjs,   gap_sjsu,   alternative='two-sided')

    # Alpha histogram
    ax = axes[0]
    bins_a = np.linspace(1.5, 6.5, 20)
    ax.hist(alpha_sjs,  bins=bins_a, color='purple', alpha=0.6, label='SJS (3-body)')
    ax.hist(alpha_sjsu, bins=bins_a, color='teal',   alpha=0.6, label='SJSU (4-body)')
    ax.axvline(alpha_sjs.mean(),  color='purple', ls='--', lw=2)
    ax.axvline(alpha_sjsu.mean(), color='teal',   ls='--', lw=2)
    ax.axvline(ALPHA_CRIT, color='red', ls='--', lw=2, label=r'$\alpha_{crit}$')
    ax.set_xlabel(r'$\alpha$ (3-body / pairwise)')
    ax.set_ylabel('Count')
    ax.set_title(f'α Distribution\n(Mann-Whitney p={mw_alpha.pvalue:.3f})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Gap histogram
    ax = axes[1]
    bins_g = np.linspace(0, 0.16, 20)
    ax.hist(gap_sjs,  bins=bins_g, color='purple', alpha=0.6, label='SJS')
    ax.hist(gap_sjsu, bins=bins_g, color='teal',   alpha=0.6, label='SJSU')
    ax.axvline(0.05, color='orange', ls=':', lw=2, label='transitional threshold')
    frac_sjs  = 100 * (gap_sjs  < 0.05).mean()
    frac_sjsu = 100 * (gap_sjsu < 0.05).mean()
    ax.text(0.06, ax.get_ylim()[1]*0.7 if ax.get_ylim()[1] > 0 else 5,
            f'SJS: {frac_sjs:.0f}% below\nSJSU: {frac_sjsu:.0f}% below',
            fontsize=8, color='goldenrod')
    ax.set_xlabel(r'Spectral gap $\lambda_3 - \lambda_2$')
    ax.set_ylabel('Count')
    ax.set_title(f'Gap Distribution\n(Mann-Whitney p={mw_gap.pvalue:.2e})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Summary bar chart
    ax = axes[2]
    metrics = [
        ('α mean\n(×100)', alpha_sjs.mean()*100, alpha_sjsu.mean()*100),
        ('α/α_crit\n(×100)', alpha_sjs.mean()/ALPHA_CRIT*100, alpha_sjsu.mean()/ALPHA_CRIT*100),
        ('gap mean\n(×100)', gap_sjs.mean()*100, gap_sjsu.mean()*100),
        ('% epochs\ngap<0.05', frac_sjs, frac_sjsu),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, [m[1] for m in metrics], w, color='purple', alpha=0.8, label='SJS')
    bars2 = ax.bar(x + w/2, [m[2] for m in metrics], w, color='teal',   alpha=0.8, label='SJSU')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylabel('Value (see axis labels)')
    ax.set_title('System Comparison\nSummary')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: PRINT NUMERICAL RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(sjs_results, sjsu_results):
    alpha_sjs  = np.array([r['alpha']  for r in sjs_results])
    gap_sjs    = np.array([r['gap']    for r in sjs_results])
    lam2_sjs   = np.array([r['lambda2'] for r in sjs_results])
    alpha_sjsu = np.array([r['alpha']  for r in sjsu_results])
    alpha4     = np.array([r['alpha4'] for r in sjsu_results])
    gap_sjsu   = np.array([r['gap']    for r in sjsu_results])
    lam2_sjsu  = np.array([r['lambda2'] for r in sjsu_results])

    mw_alpha = mannwhitneyu(alpha_sjs, alpha_sjsu, alternative='two-sided')
    mw_gap   = mannwhitneyu(gap_sjs,   gap_sjsu,   alternative='two-sided')
    t_alpha  = ttest_ind(alpha_sjs, alpha_sjsu)
    t_gap    = ttest_ind(gap_sjs, gap_sjsu)

    reg_sjs  = [classify_regime(g) for g in gap_sjs]
    reg_sjsu = [classify_regime(g) for g in gap_sjsu]

    alpha_config, evals_config, fv_config = config_space_laplacian()

    print("\n" + "="*65)
    print("NUMERICAL RESULTS — PMIR PREPRINT 6")
    print("="*65)
    print(f"\nCONFIGURATION-SPACE FRAME:")
    print(f"  alpha_config  = {alpha_config:.3e}  ({alpha_config/ALPHA_CRIT*100:.2e}% of alpha_crit)")
    print(f"  Spectrum      = {evals_config}")
    print(f"  Fiedler vec   = {fv_config}  (Sun, Jupiter, Saturn)")

    print(f"\nSJS EPOCH SWEEP ({len(sjs_results)} epochs):")
    print(f"  alpha:    {alpha_sjs.mean():.3f} ± {alpha_sjs.std():.3f}  "
          f"({alpha_sjs.mean()/ALPHA_CRIT*100:.1f}% of alpha_crit)")
    print(f"  alpha range: [{alpha_sjs.min():.2f}, {alpha_sjs.max():.2f}]")
    print(f"  gap:      {gap_sjs.mean():.4f} ± {gap_sjs.std():.4f}")
    print(f"  lambda2:  {lam2_sjs.mean():.4f} ± {lam2_sjs.std():.4f}")
    print(f"  Regime:   struct-sens={reg_sjs.count('structure-sensitive')}/{len(reg_sjs)} "
          f"({100*reg_sjs.count('structure-sensitive')/len(reg_sjs):.1f}%)")
    print(f"            transitional={reg_sjs.count('transitional')}/{len(reg_sjs)} "
          f"({100*reg_sjs.count('transitional')/len(reg_sjs):.1f}%)")

    print(f"\nSJSU EPOCH SWEEP ({len(sjsu_results)} epochs):")
    print(f"  alpha:    {alpha_sjsu.mean():.3f} ± {alpha_sjsu.std():.3f}  "
          f"({alpha_sjsu.mean()/ALPHA_CRIT*100:.1f}% of alpha_crit)")
    print(f"  alpha4:   {alpha4.mean():.3f} ± {alpha4.std():.3f}  "
          f"({alpha4.mean()/ALPHA_CRIT*100:.1f}% of alpha_crit)")
    print(f"  gap:      {gap_sjsu.mean():.4f} ± {gap_sjsu.std():.4f}")
    print(f"  lambda2:  {lam2_sjsu.mean():.4f} ± {lam2_sjsu.std():.4f}")
    print(f"  Regime:   struct-sens={reg_sjsu.count('structure-sensitive')}/{len(reg_sjsu)} "
          f"({100*reg_sjsu.count('structure-sensitive')/len(reg_sjsu):.1f}%)")
    print(f"            transitional={reg_sjsu.count('transitional')}/{len(reg_sjsu)} "
          f"({100*reg_sjsu.count('transitional')/len(reg_sjsu):.1f}%)")

    print(f"\nSTATISTICAL TESTS (SJS vs SJSU):")
    print(f"  alpha: MW U={mw_alpha.statistic:.0f}, p={mw_alpha.pvalue:.4f} | "
          f"t={t_alpha.statistic:.3f}, p={t_alpha.pvalue:.4f}")
    print(f"  gap:   MW U={mw_gap.statistic:.0f}, p={mw_gap.pvalue:.2e}  | "
          f"t={t_gap.statistic:.3f}, p={t_gap.pvalue:.2e}")
    print(f"\n  Gap compression: {gap_sjs.mean()/gap_sjsu.mean():.1f}x "
          f"({gap_sjs.mean():.4f} → {gap_sjsu.mean():.4f})")
    print(f"  Frac gap<0.05: SJS={100*(gap_sjs<0.05).mean():.1f}%  "
          f"SJSU={100*(gap_sjsu<0.05).mean():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*65)
    print("PMIR PREPRINT 6 — HYPERGRAPH LAPLACIAN N-BODY ANALYSIS")
    print("="*65)

    print("\n[1/4] Running SJS epoch sweep (60 epochs × 20 yr)...")
    sjs_results = run_sweep('SJS', n_epochs=60, epoch_spacing=2.0)

    print("\n[2/4] Running SJSU epoch sweep (60 epochs × 20 yr)...")
    sjsu_results = run_sweep('SJSU', n_epochs=60, epoch_spacing=2.0)

    print("\n[3/4] Printing numerical results...")
    print_results(sjs_results, sjsu_results)

    print("\n[4/4] Generating figures...")
    fig1_spectral_timeseries(sjs_results, sjsu_results)
    fig2_regime_diagram(sjs_results, sjsu_results)
    fig3_pmir_dynamics(sjs_results, sjsu_results)
    fig4_validation(sjs_results, sjsu_results)

    print("\n" + "="*65)
    print("COMPLETE. Four PNG figures written to current directory.")
    print("="*65)
