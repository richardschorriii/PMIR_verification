# PASS42 - Null Distribution Generation

## Purpose


Generates null distributions through permutation testing to establish
baseline comparisons for PMIR effects.

Creates randomized controls by shuffling key parameters while preserving
network structure.

Outputs:
- Null distribution statistics
- Comparison metrics
- p-values for observed effects


## Usage

```bash
python PASS42.py \
```

## Key Functions

- `ensure_dir()`
- `parse_seed_from_filename()`
- `ensure_dir()`
- `ols_r2()`

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

