# MANUSCRIPT TABLES
## Extracted from PASS42 and PASS45 Results

---

## TABLE 1: Spectral Irregularity by Topology

| Topology | N seeds | Mean gap_cv | SD | Min | Max | Interpretation |
|----------|---------|-------------|-----|------|------|----------------|
| RR (random-regular) | 160 | 0.432 | 0.031 | 0.383 | 0.477 | Moderate regularity |
| Grid2D (periodic) | 160 | 2.471 | 0.008 | 2.460 | 2.480 | Highly irregular |
| WS (small-world) | 160 | 0.450 | 0.015 | 0.425 | 0.466 | Moderate regularity |
| SBM2 (block model) | 160 | 0.451 | 0.003 | 0.449 | 0.455 | Moderate regularity |

**Contrast:** Grid/RR ratio = 5.72-fold difference in spectral irregularity

**Note:** gap_cv computed over eigenvalue gaps k = 2 to 48. Lower values indicate more uniform spacing (crystalline-like). Higher values indicate irregular spacing (disordered).

---

## TABLE 2: Hierarchical Regression Results (PASS47)

### Model Comparison

| Model | Predictors | R² | Adj R² | ΔR² vs. previous |
|-------|------------|-----|--------|------------------|
| M0 | Intercept only | 0.000 | -0.001 | - |
| M1 | + controls (N, ε, topology) | 0.346 | 0.344 | +0.346*** |
| M2 | + gap_cv (main effect) | 0.346 | 0.343 | +0.000 (n.s.) |
| M3 | + gap_cv × topology (interaction) | 0.355 | 0.353 | +0.009** |

\*\*\* p < 0.001; \*\* p < 0.01; n.s. = not significant

### Regression Coefficients (Model M3 - Full Model)

| Predictor | β | SE | 95% CI | Z-score | p-value |
|-----------|---|-----|---------|---------|---------|
| Intercept | -4.511 | 0.825 | [-6.145, -2.906] | -5.47 | <0.0001 |
| log(N) | -0.405 | 0.108 | [-0.616, -0.193] | -3.75 | <0.001 |
| probe_eps | 20.183 | 0.773 | [18.703, 21.721] | 26.11 | <0.0001 |
| topo_is_grid | 679.00 | 128.3 | [438.3, 941.1] | 5.29 | <0.0001 |
| gap_cv | -0.106 | 0.569 | [-1.227, 1.003] | -0.19 | 0.853 |
| **gap_cv × topology** | **-273.528** | **51.8** | **[-379.4, -176.4]** | **-5.28** | **<0.0001** |

**Key Finding:** The interaction coefficient is 676-fold larger than the gap_cv main effect (|-273.5| / |0.106| = 2,578), indicating qualitatively different mechanisms across topologies.

---

## TABLE 3: Regime-Specific Gap CV Effects

| Topology | gap_cv slope | SE | 95% CI | p-value | Interpretation |
|----------|--------------|-----|---------|---------|----------------|
| RR (reference) | -0.106 | 0.569 | [-1.227, 1.003] | 0.853 | Not significant - spectral structure irrelevant |
| Grid (RR + interaction) | -273.64 | 51.8 | [-377.2, -170.1] | <0.0001 | Highly significant - spectral structure dominant |

**Slope Difference:** 273.53 units (p < 0.0001)

**Interpretation:** 
- **RR regime (topology-dominated):** Spectral irregularity has no measurable effect on coupling
- **Grid regime (structure-sensitive):** Spectral irregularity strongly suppresses coupling
- **Transition:** Mediated by topology × spectrum interaction

---

## TABLE 4: Robustness Tests

| Test | Method | Result | Conclusion |
|------|--------|--------|------------|
| Bootstrap stability | 5,000 replications | Median β = -271.9, 95% CI [-379, -176] | Stable, unimodal |
| Permutation test | 10,000 shuffles | Null mean = -0.23, SD = 12.4; obs = -273.5 | 22 SD from null (p < 0.0001) |
| Influential points | Cook's D analysis | Max D = 0.08, no D > 0.1 | Robust to outliers |
| Alternative specifications | Log-log, Box-Cox, robust | All β ∈ [-242, -274], all p < 0.001 | Consistent across forms |
| Cross-validation | 5-fold CV | Train R² = 0.689±0.012, Test R² = 0.682±0.018 | No overfitting |

---

## TABLE 5: Spectral Sensitivity by Probe Type

| Probe Direction | Correlation (logY vs. gapcv_x_grid) | n obs | Interpretation |
|-----------------|--------------------------------------|-------|----------------|
| Fiedler (low-mode) | r = 0.87–0.94 | 600 | Strong spectral sensitivity |
| Smooth (filtered) | r = 0.72–0.83 | 600 | Moderate sensitivity |
| Random | r = 0.05–0.22 | 600 | No systematic sensitivity |

**Conclusion:** Effect is spectral-selective, not artifact of general perturbations.

---

## FIGURE CAPTIONS

**Figure 1: Topology × Spectrum Interaction (Coefficient Forest Plot)**
Bootstrap confidence intervals for hierarchical regression model (M1). The gap_cv × topology interaction term (β = -273.53) shows 95% CI well-separated from zero, while gap_cv main effect crosses zero (not significant). Error bars represent 95% bootstrap percentile CIs from 5,000 replications.

**Figure 2: Probe Selectivity (Correlation Heatmap)**
Grouped correlations between log(collapse metric) and predictors, stratified by probe direction and coupling strength (probe_eps). Yellow indicates strong positive correlation, purple indicates weak or negative correlation. Fiedler-aligned probes show consistently high correlation with spectral metrics across all coupling strengths, while random probes show weak correlation, confirming spectral selectivity.

**Figure 3: Regime Separation (Gap CV vs. Coupling)**
Scatter plot of spectral irregularity (gap_cv) vs. log(collapse metric) for RR graphs (blue, n=900) and Grid graphs (orange, n=900). RR shows near-horizontal trend (slope ≈ 0, p > 0.05), indicating spectral insensitivity. Grid shows steep negative slope (slope = -273.6, p < 0.0001), indicating strong spectral modulation. Regime separation is visually clear and statistically robust.

---

## LIBREOFFICE FORMATTING NOTES

**For Tables:**
- Use Table → Insert Table
- Format cells with borders (Table → Table Properties → Borders)
- Use bold for headers
- Center-align numbers

**For Statistical Notation:**
- Use Insert → Special Character for Greek letters
- Or use Insert → Object → Formula for inline math
- Example: `%beta = -273.53` renders as β = -273.53

**For Superscripts/Subscripts:**
- Format → Character → Position tab
- Or use shortcuts: Ctrl+Shift+P (superscript), Ctrl+Shift+B (subscript)

