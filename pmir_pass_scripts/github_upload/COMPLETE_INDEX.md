# Complete PASS Scripts Index

Total Scripts: **49** (PASS00 - PASS48)

## Quick Reference Table

| PASS | Features | Functions | Size |
|------|----------|-----------|------|
| PASS00 | Lambda Analysis, Topology Effects | 0 funcs | 10,857B |
| PASS01 | Lambda Analysis, Topology Effects | 0 funcs | 10,857B |
| PASS02 | Lambda Analysis, Topology Effects, Time-Lock | 0 funcs | 3,511B |
| PASS03 | Lambda Analysis, Topology Effects, Time-Lock | 3 funcs | 6,637B |
| PASS04 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 8,600B |
| PASS05 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 8,566B |
| PASS06 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 8,921B |
| PASS07 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 11,363B |
| PASS08 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 10,631B |
| PASS09 | Lambda Analysis, Fiedler/Spectral, Permutation Test | 5 funcs | 10,209B |
| PASS10 | Lambda Analysis, Fiedler/Spectral, Permutation Test | 5 funcs | 10,144B |
| PASS11 | Lambda Analysis, Fiedler/Spectral, Topology Effects | 5 funcs | 10,632B |
| PASS12 | Permutation Test, Topology Effects | 5 funcs | 8,958B |
| PASS13 | Permutation Test, Topology Effects | 5 funcs | 9,813B |
| PASS14 | Permutation Test, Topology Effects | 5 funcs | 8,745B |
| PASS15 | General | 1 funcs | 2,048B |
| PASS16 | General | 1 funcs | 2,048B |
| PASS17 | General | 1 funcs | 1,255B |
| PASS18 | General | 1 funcs | 1,255B |
| PASS19 | Permutation Test, Regression, Topology Effects | 4 funcs | 10,421B |
| PASS20 | Permutation Test, Topology Effects, Time-Lock | 3 funcs | 9,105B |
| PASS21 | Topology Effects, Time-Lock | 3 funcs | 8,724B |
| PASS22 | Topology Effects, Time-Lock | 3 funcs | 8,724B |
| PASS23 | Topology Effects, Time-Lock | 3 funcs | 8,724B |
| PASS24 | Topology Effects, Time-Lock, Null Baseline | 2 funcs | 5,971B |
| PASS25 | Topology Effects, Time-Lock, Null Baseline | 2 funcs | 5,971B |
| PASS26 | Permutation Test, Topology Effects, Time-Lock | 4 funcs | 8,632B |
| PASS27 | Permutation Test, Topology Effects | 4 funcs | 6,259B |
| PASS28 | Permutation Test, Topology Effects, Time-Lock | 3 funcs | 9,105B |
| PASS29 | Permutation Test, Topology Effects, Time-Lock | 4 funcs | 8,632B |
| PASS30 | Bootstrap SE, Topology Effects, Time-Lock | 3 funcs | 7,857B |
| PASS31 | Permutation Test, Regression, Topology Effects | 4 funcs | 10,421B |
| PASS32 | Lambda Analysis, Permutation Test, Topology Effects | 2 funcs | 7,331B |
| PASS33 | Bootstrap SE, Regression, Topology Effects | 5 funcs | 8,576B |
| PASS34 | Bootstrap SE, Regression, Topology Effects | 5 funcs | 8,604B |
| PASS35 | Regression, Topology Effects, Time-Lock | 0 funcs | 2,748B |
| PASS36 | Fiedler/Spectral, Bootstrap SE, Topology Effects | 4 funcs | 9,757B |
| PASS37 | Topology Effects | 5 funcs | 8,012B |
| PASS38 | Topology Effects | 5 funcs | 8,012B |
| PASS39 | Fiedler/Spectral, Bootstrap SE, Topology Effects | 4 funcs | 9,838B |
| PASS40 | Fiedler/Spectral, Bootstrap SE, Topology Effects | 4 funcs | 9,948B |
| PASS41 | Fiedler/Spectral, Bootstrap SE, Topology Effects | 4 funcs | 11,555B |
| PASS42 | Bootstrap SE, Regression, Topology Effects | 4 funcs | 9,339B |
| PASS43 | Bootstrap SE, Regression, Topology Effects | 4 funcs | 9,339B |
| PASS44 | Bootstrap SE, Regression, Topology Effects | 4 funcs | 9,339B |
| PASS45 | Bootstrap SE, Regression | 5 funcs | 7,395B |
| PASS46 | Bootstrap SE, Regression | 5 funcs | 5,481B |
| PASS47 | Fiedler/Spectral, Regression, Topology Effects | 0 funcs | 6,794B |
| PASS48 | Regression, Plotting, Topology Effects | 4 funcs | 4,423B |

## Detailed Script Descriptions

### PASS00.py

**Features**: Lambda Analysis, Topology Effects

- **Size**: 206 lines, 10,857 bytes
- **Input**: CSV
- **Output**: Unknown

### PASS01.py

**Features**: Lambda Analysis, Topology Effects

- **Size**: 206 lines, 10,857 bytes
- **Input**: CSV
- **Output**: Unknown

### PASS02.py

**Features**: Lambda Analysis, Topology Effects, Time-Lock

- **Size**: 83 lines, 3,511 bytes
- **Input**: CSV
- **Output**: Unknown
- **CLI Args**: Ns, degree, eps_list, graph_seeds, grid_dt

### PASS03.py

**Features**: Lambda Analysis, Topology Effects, Time-Lock

- **Size**: 199 lines, 6,637 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _to_num, fit_loglog, corr
- **CLI Args**: eps, in_csv, out_dir, pass_id

### PASS04.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 251 lines, 8,600 bytes
- **Input**: Unknown
- **Output**: CSV/Files
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Args**: N, degree, eps, n_seeds, normalize

### PASS05.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 249 lines, 8,566 bytes
- **Input**: Unknown
- **Output**: CSV/Files
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Args**: N, degree, eps, n_seeds, normalize

### PASS06.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 258 lines, 8,921 bytes
- **Input**: Unknown
- **Output**: CSV/Files
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Args**: N, degree, eps, n_seeds, normalize

### PASS07.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 305 lines, 11,363 bytes
- **Input**: Unknown
- **Output**: CSV/Files
- **Key Functions**: run, ensure_dir, read_json, find_latest_run_dir, stage_curve_for_tauhalf
- **CLI Args**: N, degree, eps, n_seeds, normalize

### PASS08.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 287 lines, 10,631 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: safe_auc, sigmoid, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Args**: features, logit_clip, median_split, mode, outdir

### PASS09.py

**Features**: Lambda Analysis, Fiedler/Spectral, Permutation Test, Topology Effects

- **Size**: 306 lines, 10,209 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: safe_auc, standardize_fit, standardize_apply, sigmoid, train_logreg_l2
- **CLI Args**: features, k, mode, outdir, seed

### PASS10.py

**Features**: Lambda Analysis, Fiedler/Spectral, Permutation Test, Topology Effects

- **Size**: 306 lines, 10,144 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: safe_auc, standardize_fit, standardize_apply, sigmoid, train_logreg_l2
- **CLI Args**: features, k, outdir, reps, seed

### PASS11.py

**Features**: Lambda Analysis, Fiedler/Spectral, Topology Effects, Time-Lock

- **Size**: 287 lines, 10,632 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: safe_auc, sigmoid, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Args**: features, logit_clip, median_split, mode, outdir

### PASS12.py

**Features**: Permutation Test, Topology Effects

- **Size**: 287 lines, 8,958 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: sigmoid, safe_corr, safe_auc, standardize_fit, standardize_apply
- **CLI Args**: base_steps, features, mode, outdir, reps

### PASS13.py

**Features**: Permutation Test, Topology Effects

- **Size**: 305 lines, 9,813 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: sigmoid, safe_auc, standardize_fit, standardize_apply, train_logreg_l2
- **CLI Args**: base_steps, features, mode, outdir, reps

### PASS14.py

**Features**: Permutation Test, Topology Effects

- **Size**: 281 lines, 8,745 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: sigmoid, safe_corr, safe_auc, standardize_fit, standardize_apply
- **CLI Args**: base_steps, features, mode, outdir, reps

### PASS15.py

- **Size**: 68 lines, 2,048 bytes
- **Input**: Unknown
- **Output**: Unknown
- **Key Functions**: best_threshold_balanced_acc

### PASS16.py

- **Size**: 68 lines, 2,048 bytes
- **Input**: Unknown
- **Output**: Unknown
- **Key Functions**: best_threshold_balanced_acc

### PASS17.py

- **Size**: 38 lines, 1,255 bytes
- **Input**: Unknown
- **Output**: Unknown
- **Key Functions**: best_bal_threshold

### PASS18.py

- **Size**: 38 lines, 1,255 bytes
- **Input**: Unknown
- **Output**: Unknown
- **Key Functions**: best_bal_threshold

### PASS19.py

**Features**: Permutation Test, Regression, Topology Effects, Time-Lock, Null Baseline

- **Size**: 267 lines, 10,421 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _safe_float, _ols_fit, _quad_fit, _perm_contrast_stratN
- **CLI Args**: in_csv, outdir, reps, score_col, seed

### PASS20.py

**Features**: Permutation Test, Topology Effects, Time-Lock, Null Baseline

- **Size**: 246 lines, 9,105 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: cohen_d, auroc_rank, stratified_permutation_pvalue
- **CLI Args**: grid_name, in_graph_csv, metrics, outdir, reps

### PASS21.py

**Features**: Topology Effects, Time-Lock

- **Size**: 251 lines, 8,724 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: add_panel, safe_auc, cohens_d
- **CLI Args**: k, match_on, outdir, panels_csv, pmir_field_csv

### PASS22.py

**Features**: Topology Effects, Time-Lock

- **Size**: 251 lines, 8,724 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: add_panel, safe_auc, cohens_d
- **CLI Args**: k, match_on, outdir, panels_csv, pmir_field_csv

### PASS23.py

**Features**: Topology Effects, Time-Lock

- **Size**: 251 lines, 8,724 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: add_panel, safe_auc, cohens_d
- **CLI Args**: k, match_on, outdir, panels_csv, pmir_field_csv

### PASS24.py

**Features**: Topology Effects, Time-Lock, Null Baseline

- **Size**: 168 lines, 5,971 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: pick_col, _n
- **CLI Args**: in_csv, k, outdir, prefer_chaotic, q

### PASS25.py

**Features**: Topology Effects, Time-Lock, Null Baseline

- **Size**: 168 lines, 5,971 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: pick_col, _n
- **CLI Args**: in_csv, k, outdir, prefer_chaotic, q

### PASS26.py

**Features**: Permutation Test, Topology Effects, Time-Lock, Null Baseline

- **Size**: 233 lines, 8,632 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN
- **CLI Args**: in_csv, metrics, outdir, reps, seed

### PASS27.py

**Features**: Permutation Test, Topology Effects

- **Size**: 180 lines, 6,259 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN
- **CLI Args**: in_csv, metrics, outdir, reps, seed

### PASS28.py

**Features**: Permutation Test, Topology Effects, Time-Lock, Null Baseline

- **Size**: 246 lines, 9,105 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: cohen_d, auroc_rank, stratified_permutation_pvalue
- **CLI Args**: grid_name, in_graph_csv, metrics, outdir, reps

### PASS29.py

**Features**: Permutation Test, Topology Effects, Time-Lock, Null Baseline

- **Size**: 233 lines, 8,632 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _resolve_metric_col, _ols_loglog, _slope_for_topology, _perm_slope_contrast_stratN
- **CLI Args**: in_csv, metrics, outdir, reps, seed

### PASS30.py

**Features**: Bootstrap SE, Topology Effects, Time-Lock, Null Baseline

- **Size**: 241 lines, 7,857 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _ols_slope, _safe_log, bootstrap_slope_stratN
- **CLI Args**: in_csv, log_floor, metrics, outdir, reps

### PASS31.py

**Features**: Permutation Test, Regression, Topology Effects, Time-Lock, Null Baseline

- **Size**: 267 lines, 10,421 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: _safe_float, _ols_fit, _quad_fit, _perm_contrast_stratN
- **CLI Args**: in_csv, outdir, reps, score_col, seed

### PASS32.py

**Features**: Lambda Analysis, Permutation Test, Topology Effects, Time-Lock

- **Size**: 191 lines, 7,331 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, perm_delta_mean_stratN
- **CLI Args**: agg, do_probe_invariance, outdir, reps, score_col

### PASS33.py

**Features**: Bootstrap SE, Regression, Topology Effects, Time-Lock

- **Size**: 232 lines, 8,576 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Args**: boot_reps, in_csv, min_eps, outdir, seed

### PASS34.py

**Features**: Bootstrap SE, Regression, Topology Effects, Time-Lock

- **Size**: 234 lines, 8,604 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, _safe_log, ols_fit, bootstrap_se, try_statsmodels_se
- **CLI Args**: boot_reps, in_csv, min_eps, outdir, seed

### PASS35.py

**Features**: Regression, Topology Effects, Time-Lock

- **Size**: 72 lines, 2,748 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **CLI Args**: agg, boot_reps, outdir, reps, score_col

### PASS36.py

**Features**: Fiedler/Spectral, Bootstrap SE, Topology Effects

- **Size**: 259 lines, 9,757 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str
- **CLI Args**: N_ref, beta_tol, cv_tol, dir_competitors, growth_tol

### PASS37.py

**Features**: Topology Effects

- **Size**: 199 lines, 8,012 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, _parse_agg_over, fit_loglinear, cv, boot_fit
- **CLI Args**: agg_over, boot_reps, in_csv, outdir, seed

### PASS38.py

**Features**: Topology Effects

- **Size**: 199 lines, 8,012 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, _parse_agg_over, fit_loglinear, cv, boot_fit
- **CLI Args**: agg_over, boot_reps, in_csv, outdir, seed

### PASS39.py

**Features**: Fiedler/Spectral, Bootstrap SE, Topology Effects

- **Size**: 261 lines, 9,838 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str
- **CLI Args**: N_ref, beta_tol, cv_tol, dir_competitors, growth_tol

### PASS40.py

**Features**: Fiedler/Spectral, Bootstrap SE, Topology Effects

- **Size**: 262 lines, 9,948 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str
- **CLI Args**: N_ref, beta_tol, cv_tol, dir_competitors, growth_tol

### PASS41.py

**Features**: Fiedler/Spectral, Bootstrap SE, Topology Effects, Time-Lock

- **Size**: 303 lines, 11,555 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, ols_xy, cv, alpha_grid_from_str
- **CLI Args**: N_ref, beta_tol, cv_tol, dir_competitors, growth_tol

### PASS42.py

**Features**: Bootstrap SE, Regression, Topology Effects, Time-Lock

- **Size**: 241 lines, 9,339 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, parse_seed_from_filename, ensure_dir, ols_r2
- **CLI Args**: boot_reps, outdir, pairs_csv, pairs_with_seed_spectral_csv, pass42_csv

### PASS43.py

**Features**: Bootstrap SE, Regression, Topology Effects, Time-Lock

- **Size**: 241 lines, 9,339 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, parse_seed_from_filename, ensure_dir, ols_r2
- **CLI Args**: boot_reps, outdir, pairs_csv, pairs_with_seed_spectral_csv, pass42_csv

### PASS44.py

**Features**: Bootstrap SE, Regression, Topology Effects, Time-Lock

- **Size**: 241 lines, 9,339 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: ensure_dir, parse_seed_from_filename, ensure_dir, ols_r2
- **CLI Args**: boot_reps, outdir, pairs_csv, pairs_with_seed_spectral_csv, pass42_csv

### PASS45.py

**Features**: Bootstrap SE, Regression

- **Size**: 218 lines, 7,395 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, build_X
- **CLI Args**: boot_reps, controls, min_rows_group, outdir, predictor

### PASS46.py

**Features**: Bootstrap SE, Regression

- **Size**: 275 lines, 5,481 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: die, safe_log, ols_fit, bootstrap_betas, build_X
- **CLI Args**: boot_reps, controls, min_rows_group, outdir, predictor

### PASS47.py

**Features**: Fiedler/Spectral, Regression, Topology Effects, Time-Lock, Null Baseline

- **Size**: 89 lines, 6,794 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **CLI Args**: boot_reps, controls, outdir, predictor, seed

### PASS48.py

**Features**: Regression, Plotting, Topology Effects

- **Size**: 141 lines, 4,423 bytes
- **Input**: CSV
- **Output**: CSV/Files
- **Key Functions**: safe_log, ols_fit, safe_log, fit_line
- **CLI Args**: controls, csv, eps, outdir, predictor

## Feature Distribution

| Feature | Count |
|---------|-------|
| Topology Effects | 43 |
| Time-Lock | 29 |
| Lambda Analysis | 13 |
| Fiedler/Spectral | 13 |
| Permutation Test | 13 |
| Regression | 12 |
| Bootstrap SE | 12 |
| Null Baseline | 10 |
| Plotting | 1 |

## Size Distribution

- **Smallest**: 1,255 bytes (PASS17)
- **Largest**: 11,555 bytes (PASS41)
- **Average**: 7,960 bytes
- **Total**: 390,087 bytes
