# Reproducibility Guide
## Complete Step-by-Step Instructions

---

## 🎯 Purpose

This guide provides complete instructions for reproducing all results in the PMIR verification study, from raw data to publication figures.

**Time Required**: 2-4 hours (depending on computational resources)

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (with WSL recommended)
- **Python**: 3.9+ (3.10 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for data and results

### Skills Required
- Basic command-line usage
- Basic Python (for understanding scripts)
- No advanced programming needed!

---

## 🚀 Setup (30 minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/richardschorriii/PMIR_verification.git
cd PMIR_verification
```

### Step 2: Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate pmir
```

**Option B: Using pip + venv**
```bash
python -m venv pmir_env
source pmir_env/bin/activate  # On Windows: pmir_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import numpy, pandas, scipy, matplotlib; print('All dependencies installed!')"
```

---

## 📊 Data Preparation (15 minutes)

### Option 1: Use Provided Processed Data (Quickest)

The repository includes pre-processed data files in `data/derived/`:
- `pass33_contrast_by_dir_probe_eps.csv`
- `pass33_summary_by_dir_probe_topoN_eps.csv`
- `pass33_by_graph_eps_dir_probe.csv`
- `pass42_eigs_band_metrics.csv`

**No additional preparation needed** - skip to "Running Analyses"!

### Option 2: Download from Zenodo (Permanent Archive)

```bash
# Download the complete dataset
wget https://zenodo.org/record/XXXXX/files/pmir_verification_data.zip
unzip pmir_verification_data.zip -d data/derived/
```

### Option 3: Generate from Raw Ephemeris (Advanced)

For full transparency, you can regenerate from JPL Horizons data:

```bash
# 1. Download ephemeris data (see data/ephemeris/README.md for instructions)
# 2. Run preprocessing pipeline (requires additional scripts)
python scripts/preprocessing/generate_phase_trajectories.py
python scripts/preprocessing/run_pmir_analysis.py
```

**Note**: This requires ~10 hours of computation. We recommend using Option 1 or 2 for verification purposes.

---

## 🔬 Running Analyses (1 hour)

### Critical Passes (Must Run)

These are the 4 critical passes that prove the breakthrough:

#### PASS34 - Scale × Coupling Interaction
```bash
python scripts/core/pass34_scaling_regression_auc.py \
    --in_csv data/derived/pass34_in_from_pass33_summary.csv \
    --outdir results/verification/pass34_verification \
    --topo_ref rr \
    --boot_reps 5000 \
    --seed 1337
```

**Expected Output**:
- `pass34_pooled_models.csv` (R² should be 0.9011)
- `pass34_per_topology_models.csv`
- `pass34_scaling_summary.txt`

**Verification**: Check that R² with interaction = 0.9011 ± 0.001

#### PASS36 - Fixed-Point Collapse
```bash
python scripts/core/pass36_fixed_point_collapse.py \
    --in_csv data/derived/pass33_summary_by_dir_probe_topoN_eps.csv \
    --outdir results/verification/pass36_verification \
    --topo_ref rr \
    --topo_alt grid2d_periodic \
    --boot_reps 5000 \
    --seed 1337
```

**Expected Output**:
- `pass36_pairs_topoN_eps.csv`
- `pass36_invariance_by_dir_mode_eps.csv` (CV range 0.26-0.88)
- `pass36_collapse_fit.csv`

#### PASS45 - Spectral Join (Preparation)
```bash
python scripts/core/pass45_seedlevel_spectral_join.py \
    --collapse_csv data/derived/pass33_by_graph_eps_dir_probe.csv \
    --spectral_csv data/derived/pass42_eigs_band_metrics.csv \
    --outdir results/verification/pass45_join \
    --collapse_metric score_mean \
    --seed_col graph_seed \
    --use_abs 1
```

**Expected Output**:
- `pass45_seedlevel_join.csv` (1800 rows, 100% join rate)
- `pass45_seedlevel_join_summary.txt`

#### PASS47 - Topology × Spectrum Interaction (**THE BREAKTHROUGH**)
```bash
python scripts/core/pass47_topology_controlled_regression.py \
    --seed_join_csv results/verification/pass45_join/pass45_seedlevel_join.csv \
    --outdir results/verification/pass47_verification \
    --predictor gap_cv \
    --topo_ref rr \
    --topo_alt grid2d_periodic \
    --boot_reps 5000 \
    --seed 1337
```

**Expected Output**:
- `pass47_topology_controlled_regression.csv`
- `pass47_group_corr.csv`
- `pass47_summary.txt`

**CRITICAL VERIFICATION**: Check the summary file for:
```
gap_cv_x_alt: β = -273.53 (should be within [-280, -267])
CI: [-379.38, -176.41] (entirely negative)
```

### Supporting Passes (Optional)

Additional robustness tests in `scripts/supporting/`:
- PASS37: Curve collapse analysis
- PASS39: Dual-space collapse
- PASS41: Eigenvalue competition
- PASS42: Spectral gap test (raw)
- PASS43: Spectral vs collapse correlation

---

## 📈 Generating Figures (30 minutes)

### PASS34 Figures
```bash
python scripts/figures/generate_pass34_figures.py \
    data/derived/pass34_in_from_pass33_summary.csv \
    results/verification/pass34_verification/pass34_pooled_models.csv \
    results/verification/pass34_verification/pass34_per_topology_models.csv \
    results/figures/pass34/
```

**Output**: 3 publication-quality PNG files (300 DPI)
- `pass34_interaction_plot.png`
- `pass34_coefficients.png`
- `pass34_r2_improvement.png`

### PASS47 Figures (**Publication Main Figures**)
```bash
python scripts/figures/generate_pass47_figures.py \
    results/verification/pass47_verification/pass47_topology_controlled_regression.csv \
    results/verification/pass45_join/pass45_seedlevel_join.csv \
    data/derived/pass42_eigs_band_metrics.csv \
    results/figures/pass47/
```

**Output**: 4 publication-quality PNG files (300 DPI)
- `pass47_forest_plot.png` (coefficient comparison)
- `pass47_interaction_effect.png` (**THE BREAKTHROUGH FIGURE**)
- `pass47_model_comparison.png`
- `pass47_spectral_distributions.png`

---

## ✅ Verification Checklist

After running all analyses, verify:

### PASS34
- [ ] R² without interaction = 0.806 ± 0.01
- [ ] R² with interaction = 0.901 ± 0.01
- [ ] Improvement = +9.5% ± 1%

### PASS36
- [ ] 180 paired comparisons
- [ ] Minimum CV ≈ 0.26
- [ ] Most CVs > 0.30 (scale-dependent)

### PASS47 (CRITICAL!)
- [ ] gap_cv × grid: β = -273.53 ± 10
- [ ] 95% CI entirely negative
- [ ] p-value < 0.0001
- [ ] R² improvement M1→M2 ≈ +0.01

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Ensure environment is activated
conda activate pmir  # or: source pmir_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### "File not found" errors
```bash
# Check you're in the repository root
pwd  # Should show: .../PMIR_verification

# Verify data files exist
ls -lh data/derived/

# If missing, download from Zenodo (see Data Preparation)
```

### Different numerical results
Small variations (< 1%) are normal due to:
- Different random number generators
- Floating-point precision
- Bootstrap sampling

If differences are >5%, check:
- [ ] Correct seed values (1337)
- [ ] Correct input files
- [ ] Python version (3.9-3.11)

### Memory errors
Reduce bootstrap replications:
```bash
# Use --boot_reps 1000 instead of 5000
# Results will be slightly less precise but faster
```

---

## 📊 Expected Runtime

On a modern laptop (4-core CPU, 16GB RAM):

| Task | Time |
|------|------|
| Environment setup | 5-10 min |
| PASS34 | 2-5 min |
| PASS36 | 5-10 min |
| PASS45 (join) | 1 min |
| PASS47 | 10-15 min |
| Figure generation | 5 min |
| **Total** | **30-50 min** |

Bootstrap with 5000 reps is the slowest step. For quick verification, use 1000 reps.

---

## 📝 Comparing Your Results

### Expected Outputs

**PASS34 Pooled Model (M3 with interaction)**:
```
R² = 0.9011
b_logN = -0.0165 ± 0.008
c_logE = 0.0315 ± 0.018
g_logNlogE = -0.0032 ± 0.002
```

**PASS47 Model M2 (with interaction)**:
```
R² = 0.3551
gap_cv_x_alt = -273.53 ± 52
CI = [-379.38, -176.41]
Z-score = -5.28
p < 0.0001
```

### Acceptable Deviations
- Coefficients: ± 5%
- R² values: ± 0.01
- Confidence intervals: ± 10%

If your results fall within these ranges: **✓ VERIFIED!**

---

## 🎓 Understanding the Results

### What PASS34 Shows
- Scale and coupling interact (not independent)
- R² improvement proves hierarchical structure
- Different from simple power-law or linear scaling

### What PASS47 Shows (**The Breakthrough**)
- Spectral structure matters for Grid but not RR
- Interaction is dominant (676× larger than main effect)
- Proves topology-dependent geometric structure

**Physical Interpretation**:
```
RR:   Newtonian regime (topology-dominated)
Grid: GR-like regime (structure-sensitive)
```

---

## 📧 Getting Help

If you encounter issues:

1. **Check this guide** - Most problems are covered
2. **Check GitHub Issues** - Others may have same problem
3. **Open new issue** - Provide:
   - Error message (full text)
   - Your command
   - System info (OS, Python version)
   - Environment setup method used

4. **Email author**: richardschorriii@gmail.com
   - Please include "PMIR Reproducibility" in subject

---

## 🎉 Success!

If you've successfully reproduced the results:

1. **Star the repository** ⭐ (helps others find it)
2. **Cite the work** (see CITATION.cff)
3. **Share your experience** (open a discussion on GitHub)

You've independently verified a scientific discovery - that's real science! 🔬

---

## 🔗 Additional Resources

- [Verification Report](VERIFICATION_REPORT.md) - Complete statistical analysis
- [Statistical Tables](STATISTICAL_TABLES.md) - All coefficients and p-values
- [Methods Documentation](METHODS.md) - Detailed methodology
- [Passes Documentation](PASSES_DOCUMENTATION.md) - All 50+ robustness passes

---

*Last updated: February 6, 2026*
*For questions: richardschorriii@gmail.com*
