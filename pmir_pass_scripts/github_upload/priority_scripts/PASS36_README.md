# PASS36 - Direction-Mode Cross-Topology Comparison

## Purpose


Compares directional probes (Fiedler, smooth, random) across different
network topologies (random regular vs. periodic 2D grid).

Tests topology-dependent effects by analyzing how probe directions
interact with network structure.

Outputs:
- Alpha-blend optimization results
- Cross-topology effect sizes
- Statistical significance tests


## Usage

```bash
python PASS36.py \
  --in_csv <value> \
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--in_csv` | PASS33 summary_by_dir_probe_topoN_eps.csv |

## Key Functions

- `ensure_dir()`
- `ols_xy()`
- `cv()`
- `alpha_grid_from_str()`

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

- PASS34: Growth Exponent Regression
- PASS45: Time-Lock Comprehensive Analysis

