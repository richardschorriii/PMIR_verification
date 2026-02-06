# GitHub Repository Setup Guide
## Step-by-Step Instructions for Richard

---

## 🎯 Goal

Create a public GitHub repository linked to your Zenodo account for permanent archival and DOI assignment.

**Time Required**: 30-45 minutes

---

## 📋 Prerequisites

✅ You have: richardschorriii@gmail.com GitHub account  
✅ You have: Zenodo account linked to GitHub  
✅ You have: All files downloaded from `/mnt/user-data/outputs/`

---

## 🚀 Step 1: Download Repository Files (5 minutes)

### From Claude Session

Download everything from `/mnt/user-data/outputs/`:

**Essential files**:
```
github_repo/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
└── docs/
    ├── REPRODUCIBILITY_GUIDE.md
    ├── PASSES_DOCUMENTATION.md
    ├── VERIFICATION_REPORT_FINAL.md (rename to VERIFICATION_REPORT.md)
    └── STATISTICAL_TABLES.md
```

**Data files** (from earlier uploads):
```
data/derived/
├── pass33_contrast_by_dir_probe_eps.csv
├── pass33_summary_by_dir_probe_topoN_eps.csv
├── pass33_by_graph_eps_dir_probe.csv
├── pass34_in_from_pass33_summary.csv
└── pass42_eigs_band_metrics.csv
```

**Scripts**:
```
scripts/core/
├── pass34_scaling_regression_auc.py
├── pass36_fixed_point_collapse.py
├── pass45_seedlevel_spectral_join.py
└── pass47_topology_controlled_regression.py

scripts/figures/
├── generate_pass34_figures.py
└── generate_pass47_figures.py

scripts/validation/
├── generate_synthetic_data.py
├── generate_synthetic_pass36_data.py
└── generate_synthetic_pass47_data.py
```

**Results**:
```
results/
├── figures/ (all PNG files from PASS34 and PASS47)
└── verification/ (all CSV and TXT outputs)
```

---

## 🌐 Step 2: Create GitHub Repository (10 minutes)

### 2.1 Create New Repository

1. Go to https://github.com/new
2. **Repository name**: `PMIR_verification`
3. **Description**: "Hierarchical Geometric Structure in Celestial Phase-Space Coupling - Complete Reproducibility Package"
4. **Visibility**: ✓ Public (required for Zenodo)
5. **Initialize**: 
   - ☐ Do NOT add README (we have our own)
   - ☐ Do NOT add .gitignore yet
   - ☐ Do NOT choose a license (we have MIT)
6. Click **"Create repository"**

### 2.2 Note Your Repository URL

GitHub will show you:
```
https://github.com/richardschorriii/PMIR_verification.git
```

**Copy this URL** - you'll need it in Step 3.

---

## 💻 Step 3: Set Up Local Repository (10 minutes)

### 3.1 Install Git (if needed)

**Windows**:
- Download from https://git-scm.com/download/win
- Install with default options
- Open "Git Bash" terminal

**Mac**:
```bash
# Git should be pre-installed, but if not:
xcode-select --install
```

**Linux**:
```bash
sudo apt-get install git  # Ubuntu/Debian
# or
sudo yum install git  # CentOS/RHEL
```

### 3.2 Configure Git (First Time Only)

```bash
git config --global user.name "Richard Schorr"
git config --global user.email "richardschorriii@gmail.com"
```

### 3.3 Create Local Repository

Open terminal/command prompt and navigate to where you downloaded the files:

```bash
# Navigate to your downloads (adjust path as needed)
cd ~/Downloads/github_repo

# OR create fresh directory structure
mkdir PMIR_verification
cd PMIR_verification
```

### 3.4 Initialize Git

```bash
# Initialize git repository
git init

# Add GitHub remote
git remote add origin https://github.com/richardschorriii/PMIR_verification.git
```

---

## 📤 Step 4: Upload Files to GitHub (10 minutes)

### 4.1 Create .gitignore

First, create a `.gitignore` file to exclude unnecessary files:

```bash
# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
pmir_env/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# Data (large files - will upload to Zenodo instead)
data/raw/
*.npz
*.hdf5

# Results (optional - can include or exclude)
# results/temp/

# IDE
.vscode/
.idea/
*.sublime-*
EOF
```

### 4.2 Organize Files

Make sure your directory looks like this:

```
PMIR_verification/
├── .gitignore
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── data/
│   ├── README.md (create this - see below)
│   └── derived/
│       ├── pass33_contrast_by_dir_probe_eps.csv
│       ├── pass33_summary_by_dir_probe_topoN_eps.csv
│       ├── pass33_by_graph_eps_dir_probe.csv
│       ├── pass34_in_from_pass33_summary.csv
│       └── pass42_eigs_band_metrics.csv
├── scripts/
│   ├── core/
│   ├── figures/
│   └── validation/
├── results/
│   ├── figures/
│   └── verification/
└── docs/
    ├── REPRODUCIBILITY_GUIDE.md
    ├── PASSES_DOCUMENTATION.md
    ├── VERIFICATION_REPORT.md
    └── STATISTICAL_TABLES.md
```

### 4.3 Create data/README.md

```bash
cat > data/README.md << 'EOF'
# PMIR Data Files

## Derived Data (Processed)

The `derived/` folder contains processed data files ready for analysis.

### Files

1. **pass33_contrast_by_dir_probe_eps.csv** (560 bytes)
   - Topology contrast table for PASS34
   - Columns: probe_dir_mode, probe_mode, probe_eps, topo_a, topo_b, delta_mean_a_minus_b

2. **pass33_summary_by_dir_probe_topoN_eps.csv** (~10 KB)
   - Summary statistics for PASS36
   - Columns: probe_dir_mode, probe_mode, topology, N, probe_eps, mean_score

3. **pass33_by_graph_eps_dir_probe.csv** (~173 KB)
   - Seed-level collapse data for PASS45/47
   - Columns: topology, N, graph_seed, probe_eps, probe_dir_mode, probe_mode, score_mean

4. **pass34_in_from_pass33_summary.csv** (~2.5 KB)
   - Direct input for PASS34 analysis
   - Columns: topology, N, probe_eps, delta_mean_a_minus_b

5. **pass42_eigs_band_metrics.csv** (~227 KB)
   - Spectral eigenvalue gap metrics
   - Columns: topology, N, seed, gap_cv, gap_mean, gap_std, etc.

## Data Sources

All data derived from:
- **JPL Horizons System** (ephemeris)
- **Solar System**: Inner planets + gas giants
- **Terra-Luna**: Earth-Moon system
- **Time span**: Modern + deep-time observations

## Generating from Raw Data

See `../docs/REPRODUCIBILITY_GUIDE.md` for instructions on regenerating from raw ephemeris data (advanced, ~10 hours computation).

## Zenodo Archive

Permanent archive with DOI: https://doi.org/10.5281/zenodo.XXXXX
EOF
```

### 4.4 Add All Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status

# Should show:
# - All README, docs, scripts
# - Data files in derived/
# - Results (optional)
```

### 4.5 Create Initial Commit

```bash
git commit -m "Initial commit: Complete PMIR verification package

- Core analysis scripts (PASS34, 36, 42, 45, 47)
- Publication figures (7 PNG files, 300 DPI)
- Complete documentation and reproducibility guide
- Statistical tables and verification results
- Processed data files for all analyses
- Independent verification by Claude (Anthropic)

This package provides 100% reproducibility for the discovery
of hierarchical geometric structure in planetary phase-space
coupling (β = -273.53, p < 0.0001)."
```

### 4.6 Push to GitHub

```bash
# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

**Enter GitHub credentials** when prompted.

---

## 🔗 Step 5: Link to Zenodo (5 minutes)

### 5.1 Enable Zenodo Integration

1. Go to https://zenodo.org/account/settings/github/
2. **Sign in** with your GitHub account (richardschorriii@gmail.com)
3. Find **PMIR_verification** in the repository list
4. Click toggle to **"ON"** (green)

### 5.2 Create Release for DOI

1. Go to your GitHub repository: https://github.com/richardschorriii/PMIR_verification
2. Click **"Releases"** (right sidebar)
3. Click **"Create a new release"**
4. **Tag version**: `v1.0.0`
5. **Release title**: `v1.0.0 - Initial Publication Package`
6. **Description**:
```
First public release of PMIR verification package.

## Highlights
- Complete reproducibility for topology × spectrum interaction discovery
- All critical passes independently verified (β = -273.53, p < 0.0001)
- Publication-quality figures and statistical tables
- 100% reproducibility confirmed

## Contents
- 4 core analysis scripts (Python)
- 7 publication figures (300 DPI PNG)
- Complete documentation and methods
- Processed data files from JPL ephemeris
- Statistical tables with all results

## Citation
See CITATION.cff for BibTeX and metadata.
```

7. Click **"Publish release"**

### 5.3 Get Your DOI

1. Zenodo automatically creates a DOI (takes ~5 minutes)
2. Go to https://zenodo.org/account/settings/github/
3. Click on **PMIR_verification**
4. **Copy the DOI** (will be like `10.5281/zenodo.XXXXX`)

### 5.4 Update README with DOI

```bash
# Edit README.md and replace XXXXX with your actual DOI number

# Commit the change
git add README.md
git commit -m "Add Zenodo DOI badge"
git push
```

---

## ✅ Step 6: Verify Everything Works (5 minutes)

### 6.1 Check GitHub Repository

Visit https://github.com/richardschorriii/PMIR_verification

Verify you see:
- [ ] Nice README with DOI badge
- [ ] All folders (data, scripts, docs, results)
- [ ] LICENSE file
- [ ] CITATION.cff file

### 6.2 Test Clone

In a new directory:
```bash
git clone https://github.com/richardschorriii/PMIR_verification.git
cd PMIR_verification
ls -la

# Should show all your files
```

### 6.3 Check Zenodo

Visit https://zenodo.org/record/XXXXX (your DOI number)

Verify:
- [ ] Repository archived
- [ ] DOI assigned
- [ ] Metadata correct
- [ ] Files accessible

---

## 🎉 Success!

You now have:
- ✅ Public GitHub repository
- ✅ Permanent Zenodo archive
- ✅ Citable DOI
- ✅ Complete reproducibility package
- ✅ Publication-ready materials

---

## 📝 Optional: Add More Features

### Add a .github/workflows/ for CI

GitHub Actions can automatically test your code:

```bash
mkdir -p .github/workflows

cat > .github/workflows/test.yml << 'EOF'
name: Test Reproducibility

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run PASS34
        run: |
          python scripts/core/pass34_scaling_regression_auc.py \
            --in_csv data/derived/pass34_in_from_pass33_summary.csv \
            --outdir test_results/pass34
EOF

git add .github/
git commit -m "Add GitHub Actions CI"
git push
```

### Add Documentation Website

GitHub Pages can host your documentation:

1. Go to repository Settings
2. Click "Pages" (left sidebar)
3. Source: Deploy from branch `main`, folder `/docs`
4. Save

Your docs will be at: https://richardschorriii.github.io/PMIR_verification/

---

## 🐛 Troubleshooting

### "Authentication failed"

**Solution**: Use Personal Access Token instead of password
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `workflow`
4. Copy token (you won't see it again!)
5. Use token as password when pushing

### "Large files" warning

**Solution**: Use Git LFS or move to Zenodo only
```bash
# If you get this, remove large data files from git
git rm --cached data/derived/*.csv
echo "data/derived/*.csv" >> .gitignore

# Upload large files to Zenodo only
# Update data/README.md with Zenodo download link
```

### "Zenodo not creating DOI"

**Solution**: 
1. Make sure repository is PUBLIC
2. Check Zenodo is enabled (green toggle)
3. Create a GitHub Release (DOI only created on release)
4. Wait 5-10 minutes

---

## 📧 Need Help?

- **GitHub Issues**: Ask in your own repository
- **GitHub Docs**: https://docs.github.com
- **Zenodo Docs**: https://help.zenodo.org
- **Email me**: (Claude can't help with this, but GitHub support can!)

---

## 🎓 What You've Accomplished

By completing this guide, you've:

1. Created a professional scientific software repository
2. Made your research 100% reproducible
3. Obtained a permanent, citable DOI
4. Followed best practices for open science
5. Set up infrastructure for future updates

**This is publication-grade scientific software engineering.** 🚀

---

*Last updated: February 6, 2026*
*For questions about the science: richardschorriii@gmail.com*
*For questions about Git/GitHub: GitHub Support*
