# PASS45 - Time-Lock Comprehensive Analysis

## Purpose


Comprehensive analysis of time-lock structures in rotational dynamics.
Examines when and how quickly rotational patterns emerge.

Tests the hypothesis that topology constrains the temporal evolution
of rotation.

Outputs:
- Time-to-lock distributions
- Lock strength metrics
- Topology-stratified results


## Usage

```bash
python PASS45.py \
  --seed_join_csv <value> \
  --predictor <value> \
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--seed_join_csv` | pass45_seedlevel_join.csv |
| `--predictor` | primary predictor (default gap_cv) |

## Key Functions

- `ols_fit()`
- `build_X()`

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
- PASS47: Final Verification Suite

