# PMIR Verification Tests (PASS Scripts)

Complete collection of 49 Python verification tests for the PMIR (Permutation-Mean Intrinsic Rotation) framework.

## Overview

These scripts verify various aspects of rotational dynamics in networks, testing hypotheses about topology effects, characteristic dimensions, and time-lock structures.

## Quick Reference

### Priority PASS Scripts (Mentioned in Manuscript)

- **PASS34**: Topology comparison
- **PASS36**: Fiedler vector/spectral analysis
- **PASS42**: Topology comparison
- **PASS45**: Statistical bootstrap analysis
- **PASS47**: Fiedler vector/spectral analysis

## Scripts by Category

### Topology Analysis

#### PASS36 - PASS36.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,757 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS39 - PASS39.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,838 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS40 - PASS40.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,948 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS41 - PASS41.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 11,555 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS47 - PASS47.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 6,794 bytes

### Lambda Analysis

#### PASS00 - PASS00.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,857 bytes
- **Data Processing**: Uses pandas DataFrames

#### PASS01 - PASS01.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,857 bytes
- **Data Processing**: Uses pandas DataFrames

#### PASS02 - PASS02.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 3,511 bytes
- **Data Processing**: Uses pandas DataFrames

#### PASS03 - PASS03.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 6,637 bytes
- **Key Functions**: _to_num, fit_loglog, corr, main
- **CLI Arguments**: 4 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS04 - PASS04.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 8,600 bytes
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Arguments**: 20 parameters

#### PASS05 - PASS05.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 8,566 bytes
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Arguments**: 20 parameters

#### PASS06 - PASS06.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 8,921 bytes
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Arguments**: 20 parameters

#### PASS07 - PASS07.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 11,363 bytes
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Arguments**: 20 parameters

#### PASS08 - PASS08.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,631 bytes
- **Key Functions**: safe_auc, sigmoid, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Arguments**: 5 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS09 - PASS09.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,209 bytes
- **Key Functions**: safe_auc, standardize_fit, standardize_apply, sigmoid, train_logreg_l2
- **CLI Arguments**: 9 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS10 - PASS10.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,144 bytes
- **Key Functions**: safe_auc, standardize_fit, standardize_apply, sigmoid, train_logreg_l2
- **CLI Arguments**: 8 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS11 - PASS11.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 10,632 bytes
- **Key Functions**: safe_auc, sigmoid, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Arguments**: 5 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS32 - PASS32.py
- **Purpose**: Lambda (characteristic dimension) analysis
- **Size**: 7,331 bytes
- **Key Functions**: ensure_dir, perm_delta_mean_stratN, main
- **CLI Arguments**: 8 parameters
- **Data Processing**: Uses pandas DataFrames

### Visualization

#### PASS48 - PASS48.py
- **Purpose**: Topology comparison
- **Size**: 4,423 bytes
- **Key Functions**: safe_log, ols_fit, main, safe_log, fit_line
- **CLI Arguments**: 12 parameters
- **Data Processing**: Uses pandas DataFrames
- **Visualization**: Generates plots

### Regression

#### PASS19 - PASS19.py
- **Purpose**: Topology comparison
- **Size**: 10,421 bytes
- **Key Functions**: _safe_float, _ols_fit, _quad_fit, _perm_contrast_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS26 - PASS26.py
- **Purpose**: Topology comparison
- **Size**: 8,632 bytes
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS27 - PASS27.py
- **Purpose**: Topology comparison
- **Size**: 6,259 bytes
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS29 - PASS29.py
- **Purpose**: Topology comparison
- **Size**: 8,632 bytes
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS30 - PASS30.py
- **Purpose**: Topology comparison
- **Size**: 7,857 bytes
- **Key Functions**: _ols_slope, _safe_log, bootstrap_slope_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS31 - PASS31.py
- **Purpose**: Topology comparison
- **Size**: 10,421 bytes
- **Key Functions**: _safe_float, _ols_fit, _quad_fit, _perm_contrast_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS33 - PASS33.py
- **Purpose**: Topology comparison
- **Size**: 8,576 bytes
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS34 - PASS34.py
- **Purpose**: Topology comparison
- **Size**: 8,604 bytes
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS36 - PASS36.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,757 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS39 - PASS39.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,838 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS40 - PASS40.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 9,948 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS41 - PASS41.py
- **Purpose**: Fiedler vector/spectral analysis
- **Size**: 11,555 bytes
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str, main
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS42 - PASS42.py
- **Purpose**: Topology comparison
- **Size**: 9,339 bytes
- **Key Functions**: ensure_dir, parse_seed_from_filename, main, ensure_dir, ols_r2
- **CLI Arguments**: 11 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS43 - PASS43.py
- **Purpose**: Topology comparison
- **Size**: 9,339 bytes
- **Key Functions**: ensure_dir, parse_seed_from_filename, main, ensure_dir, ols_r2
- **CLI Arguments**: 11 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS44 - PASS44.py
- **Purpose**: Topology comparison
- **Size**: 9,339 bytes
- **Key Functions**: ensure_dir, parse_seed_from_filename, main, ensure_dir, ols_r2
- **CLI Arguments**: 11 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS45 - PASS45.py
- **Purpose**: Statistical bootstrap analysis
- **Size**: 7,395 bytes
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, main
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS46 - PASS46.py
- **Purpose**: Statistical bootstrap analysis
- **Size**: 5,481 bytes
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS48 - PASS48.py
- **Purpose**: Topology comparison
- **Size**: 4,423 bytes
- **Key Functions**: safe_log, ols_fit, main, safe_log, fit_line
- **CLI Arguments**: 12 parameters
- **Data Processing**: Uses pandas DataFrames
- **Visualization**: Generates plots

### Bootstrap/Resampling

#### PASS30 - PASS30.py
- **Purpose**: Topology comparison
- **Size**: 7,857 bytes
- **Key Functions**: _ols_slope, _safe_log, bootstrap_slope_stratN, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS33 - PASS33.py
- **Purpose**: Topology comparison
- **Size**: 8,576 bytes
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS34 - PASS34.py
- **Purpose**: Topology comparison
- **Size**: 8,604 bytes
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS45 - PASS45.py
- **Purpose**: Statistical bootstrap analysis
- **Size**: 7,395 bytes
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, main
- **CLI Arguments**: 6 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS46 - PASS46.py
- **Purpose**: Statistical bootstrap analysis
- **Size**: 5,481 bytes
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

### Other

#### PASS12 - PASS12.py
- **Purpose**: Topology comparison
- **Size**: 8,958 bytes
- **Key Functions**: sigmoid, safe_corr, safe_auc, standardize_fit, standardize_apply
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS13 - PASS13.py
- **Purpose**: Topology comparison
- **Size**: 9,813 bytes
- **Key Functions**: sigmoid, safe_auc, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Arguments**: 12 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS14 - PASS14.py
- **Purpose**: Topology comparison
- **Size**: 8,745 bytes
- **Key Functions**: sigmoid, safe_corr, safe_auc, standardize_fit, standardize_apply
- **CLI Arguments**: 14 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS15 - PASS15.py
- **Purpose**: General analysis
- **Size**: 2,048 bytes
- **Key Functions**: best_threshold_balanced_acc

#### PASS16 - PASS16.py
- **Purpose**: General analysis
- **Size**: 2,048 bytes
- **Key Functions**: best_threshold_balanced_acc

#### PASS17 - PASS17.py
- **Purpose**: General analysis
- **Size**: 1,255 bytes
- **Key Functions**: best_bal_threshold

#### PASS18 - PASS18.py
- **Purpose**: General analysis
- **Size**: 1,255 bytes
- **Key Functions**: best_bal_threshold

#### PASS20 - PASS20.py
- **Purpose**: Topology comparison
- **Size**: 9,105 bytes
- **Key Functions**: cohen_d, auroc_rank, stratified_permutation_pvalue, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS21 - PASS21.py
- **Purpose**: Topology comparison
- **Size**: 8,724 bytes
- **Key Functions**: main, add_panel, safe_auc, cohens_d, main
- **CLI Arguments**: 9 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS22 - PASS22.py
- **Purpose**: Topology comparison
- **Size**: 8,724 bytes
- **Key Functions**: main, add_panel, safe_auc, cohens_d, main
- **CLI Arguments**: 9 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS23 - PASS23.py
- **Purpose**: Topology comparison
- **Size**: 8,724 bytes
- **Key Functions**: main, add_panel, safe_auc, cohens_d, main
- **CLI Arguments**: 9 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS24 - PASS24.py
- **Purpose**: Topology comparison
- **Size**: 5,971 bytes
- **Key Functions**: pick_col, main, _n
- **CLI Arguments**: 5 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS25 - PASS25.py
- **Purpose**: Topology comparison
- **Size**: 5,971 bytes
- **Key Functions**: pick_col, main, _n
- **CLI Arguments**: 5 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS28 - PASS28.py
- **Purpose**: Topology comparison
- **Size**: 9,105 bytes
- **Key Functions**: cohen_d, auroc_rank, stratified_permutation_pvalue, main
- **CLI Arguments**: 7 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS35 - PASS35.py
- **Purpose**: Topology comparison
- **Size**: 2,748 bytes
- **Data Processing**: Uses pandas DataFrames

#### PASS37 - PASS37.py
- **Purpose**: Topology comparison
- **Size**: 8,012 bytes
- **Key Functions**: ensure_dir, _parse_agg_over, fit_loglinear, main, cv
- **CLI Arguments**: 8 parameters
- **Data Processing**: Uses pandas DataFrames

#### PASS38 - PASS38.py
- **Purpose**: Topology comparison
- **Size**: 8,012 bytes
- **Key Functions**: ensure_dir, _parse_agg_over, fit_loglinear, main, cv
- **CLI Arguments**: 8 parameters
- **Data Processing**: Uses pandas DataFrames

## Complete Script Index

| PASS | Purpose | Size | Functions |
|------|---------|------|----------|
| PASS00 | Lambda (characteristic dimension) analysis | 10,857B | 0 |
| PASS01 | Lambda (characteristic dimension) analysis | 10,857B | 0 |
| PASS02 | Lambda (characteristic dimension) analysis | 3,511B | 0 |
| PASS03 | Lambda (characteristic dimension) analysis | 6,637B | 4 |
| PASS04 | Lambda (characteristic dimension) analysis | 8,600B | 13 |
| PASS05 | Lambda (characteristic dimension) analysis | 8,566B | 13 |
| PASS06 | Lambda (characteristic dimension) analysis | 8,921B | 13 |
| PASS07 | Lambda (characteristic dimension) analysis | 11,363B | 13 |
| PASS08 | Lambda (characteristic dimension) analysis | 10,631B | 12 |
| PASS09 | Lambda (characteristic dimension) analysis | 10,209B | 18 |
| PASS10 | Lambda (characteristic dimension) analysis | 10,144B | 19 |
| PASS11 | Lambda (characteristic dimension) analysis | 10,632B | 12 |
| PASS12 | Topology comparison | 8,958B | 17 |
| PASS13 | Topology comparison | 9,813B | 17 |
| PASS14 | Topology comparison | 8,745B | 17 |
| PASS15 | General analysis | 2,048B | 1 |
| PASS16 | General analysis | 2,048B | 1 |
| PASS17 | General analysis | 1,255B | 1 |
| PASS18 | General analysis | 1,255B | 1 |
| PASS19 | Topology comparison | 10,421B | 5 |
| PASS20 | Topology comparison | 9,105B | 4 |
| PASS21 | Topology comparison | 8,724B | 5 |
| PASS22 | Topology comparison | 8,724B | 5 |
| PASS23 | Topology comparison | 8,724B | 5 |
| PASS24 | Topology comparison | 5,971B | 3 |
| PASS25 | Topology comparison | 5,971B | 3 |
| PASS26 | Topology comparison | 8,632B | 5 |
| PASS27 | Topology comparison | 6,259B | 5 |
| PASS28 | Topology comparison | 9,105B | 4 |
| PASS29 | Topology comparison | 8,632B | 5 |
| PASS30 | Topology comparison | 7,857B | 4 |
| PASS31 | Topology comparison | 10,421B | 5 |
| PASS32 | Lambda (characteristic dimension) analysis | 7,331B | 3 |
| PASS33 | Topology comparison | 8,576B | 8 |
| PASS34 | Topology comparison | 8,604B | 8 |
| PASS35 | Topology comparison | 2,748B | 0 |
| PASS36 | Fiedler vector/spectral analysis | 9,757B | 5 |
| PASS37 | Topology comparison | 8,012B | 6 |
| PASS38 | Topology comparison | 8,012B | 6 |
| PASS39 | Fiedler vector/spectral analysis | 9,838B | 5 |
| PASS40 | Fiedler vector/spectral analysis | 9,948B | 5 |
| PASS41 | Fiedler vector/spectral analysis | 11,555B | 5 |
| PASS42 | Topology comparison | 9,339B | 6 |
| PASS43 | Topology comparison | 9,339B | 6 |
| PASS44 | Topology comparison | 9,339B | 6 |
| PASS45 | Statistical bootstrap analysis | 7,395B | 6 |
| PASS46 | Statistical bootstrap analysis | 5,481B | 6 |
| PASS47 | Fiedler vector/spectral analysis | 6,794B | 0 |
| PASS48 | Topology comparison | 4,423B | 6 |

## Usage

Most PASS scripts follow this pattern:

```bash
python PASSXX.py --in_csv <data.csv> --outdir <results/> [options]
```

Common arguments:
- `--in_csv`: Input CSV file with PMIR data
- `--outdir`: Output directory for results
- `--topology`: Network topology (rr, grid2d_periodic, etc.)
- `--N`: Network size
- `--seed`: Random seed for reproducibility

## Dependencies

Common dependencies across all scripts:
- Python 3.7+
- NumPy
- pandas
- matplotlib (for visualization scripts)
- scipy (for some statistical tests)

## Citation

If you use these verification tests, please cite the PMIR manuscript.

