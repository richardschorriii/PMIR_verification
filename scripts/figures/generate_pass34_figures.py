#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for PASS34 (Scale × Coupling Interaction)

Creates:
1. Interaction plot: Δ vs log(N) colored by ε, separate panels for topologies
2. Coefficient comparison: RR vs Grid scaling laws
3. R² improvement bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_pass34_figures(input_csv, pooled_csv, per_topo_csv, outdir):
    """Generate all PASS34 figures"""
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_csv)
    pooled = pd.read_csv(pooled_csv)
    per_topo = pd.read_csv(per_topo_csv)
    
    # Figure 1: Interaction Plot (2 panels side by side)
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)
    
    # Color map for epsilon values
    eps_values = sorted(df['probe_eps'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eps_values)))
    
    for idx, topo in enumerate(['rr', 'grid2d_periodic']):
        ax = fig.add_subplot(gs[0, idx])
        sub = df[df['topology'] == topo].copy()
        
        # Plot each epsilon separately
        for i, eps in enumerate(eps_values):
            mask = sub['probe_eps'] == eps
            x = np.log(sub.loc[mask, 'N'])
            y = sub.loc[mask, 'delta_mean_a_minus_b']
            
            ax.scatter(x, y, c=[colors[i]], label=f'ε={eps:.3f}', 
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Fit line for this epsilon
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), c=colors[i], alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('log(N)', fontweight='bold')
        ax.set_ylabel('Δ (Grid - RR)', fontweight='bold')
        ax.set_title(f'{"Random-Regular" if topo == "rr" else "2D Periodic Grid"}',
                    fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     title='Coupling ε', frameon=True, fancybox=True)
    
    fig.suptitle('PASS34: Scale × Coupling Interaction Effect', 
                fontweight='bold', fontsize=14, y=1.02)
    
    fig.savefig(outdir / 'pass34_interaction_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Coefficient Comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Extract coefficients for with_cross model
    per_cross = per_topo[per_topo['model'] == 'with_cross'].copy()
    
    coef_names = ['b_logN', 'c_logE', 'd_logNlogE']
    coef_labels = ['β (log N)', 'γ (log ε)', 'δ (log N × log ε)']
    
    for i, (coef, label) in enumerate(zip(coef_names, coef_labels)):
        ax = axes[i]
        
        topos = per_cross['topology'].values
        vals = per_cross[coef].values
        errs = per_cross[f'{coef}_se'].values
        
        x_pos = np.arange(len(topos))
        colors_bar = ['#2E86AB', '#A23B72']
        
        bars = ax.bar(x_pos, vals, yerr=errs, capsize=5, 
                     color=colors_bar, edgecolor='black', linewidth=1.5,
                     error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['RR', 'Grid'], fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title(label, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, vals)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=9, fontweight='bold')
    
    fig.suptitle('PASS34: Scaling Law Coefficients by Topology', 
                fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / 'pass34_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: R² Improvement
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = pooled['model'].values
    r2_vals = pooled['r2'].values
    
    x_pos = np.arange(len(models))
    colors_r2 = ['#95B8D1', '#E76F51']
    
    bars = ax.bar(x_pos, r2_vals, color=colors_r2, edgecolor='black', 
                 linewidth=2, width=0.6)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No Interaction', 'With Interaction'], 
                       fontweight='bold', fontsize=11)
    ax.set_ylabel('R² (Variance Explained)', fontweight='bold', fontsize=12)
    ax.set_title('PASS34: Model Performance Comparison', 
                fontweight='bold', fontsize=14, pad=15)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels and improvement annotation
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.4f}',
               ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Add improvement arrow and text
    improvement = r2_vals[1] - r2_vals[0]
    ax.annotate('', xy=(1, r2_vals[1]), xytext=(0, r2_vals[0]),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, (r2_vals[0] + r2_vals[1])/2,
           f'+{improvement:.3f}\n({improvement*100:.1f}% improvement)',
           ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
           fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(outdir / 'pass34_r2_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[PASS34 FIGURES] Generated 3 publication-quality figures in {outdir}")
    print(f"  - pass34_interaction_plot.png")
    print(f"  - pass34_coefficients.png")
    print(f"  - pass34_r2_improvement.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python generate_pass34_figures.py <input_csv> <pooled_csv> <per_topo_csv> <outdir>")
        sys.exit(1)
    
    create_pass34_figures(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
