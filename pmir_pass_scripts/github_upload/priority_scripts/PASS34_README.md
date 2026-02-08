# PASS34 - Growth Exponent Regression

## Purpose


Tests whether characteristic dimension (lambda) follows a power-law 
relationship with network size N. Uses OLS regression with bootstrap 
standard errors to estimate growth exponents.

Key hypothesis: λ ~ N^β where β is the growth exponent.

Outputs:
- Regression coefficients and R² values
- Bootstrap confidence intervals
- Goodness-of-fit diagnostics


## Usage

```bash
python PASS34.py \
  --in_csv <value> \
  --topo_ref <value> \
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--in_csv` | PASS33 contrast table: topology,N,probe_eps,delta_mean_a_minus_b,... |
| `--topo_ref` | reference topology for pooled dummy encoding |

## Key Functions

- `ols_fit()`
- `try_statsmodels_se()`
- `build_design()`
- `per_topology_fit()`

## Input Format

Expected input CSV should contain:
- Network topology identifiers
- Size parameter (N)
- PMIR measurements
- Additional experiment parameters

## Output

Results are saved to the specified output directory:
- Summary statistics (CSV)
- Detailed results (CSV)
- Optional visualization plots (PNG)

## Related PASS Scripts

- PASS36: Direction-Mode Cross-Topology Comparison
- PASS42: Null Distribution Generation

