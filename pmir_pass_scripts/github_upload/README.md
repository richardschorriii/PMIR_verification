# PMIR Verification Tests (PASS Scripts)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)

Complete collection of **49 Python verification tests** for the PMIR (Permutation-Mean Intrinsic Rotation) framework.

## 📖 Overview

These scripts systematically test the PMIR framework's predictions about:

- **Topology effects**: How network structure influences rotational dynamics
- **Characteristic dimensions**: Power-law relationships between λ and network size
- **Time-lock structures**: Temporal emergence of rotational patterns
- **Null baselines**: Permutation-based controls for hypothesis testing

## 🎯 Priority Scripts (Cited in Manuscript)

These five scripts contain the core verification tests:

### [`PASS34`](PASS34.py) - Growth Exponent Regression
Tests whether characteristic dimension (lambda) follows a power-law 

📄 [Detailed Documentation](PASS34_README.md)

### [`PASS36`](PASS36.py) - Direction-Mode Cross-Topology Comparison
Compares directional probes (Fiedler, smooth, random) across different

📄 [Detailed Documentation](PASS36_README.md)

### [`PASS42`](PASS42.py) - Null Distribution Generation
Generates null distributions through permutation testing to establish

📄 [Detailed Documentation](PASS42_README.md)

### [`PASS45`](PASS45.py) - Time-Lock Comprehensive Analysis
Comprehensive analysis of time-lock structures in rotational dynamics.

📄 [Detailed Documentation](PASS45_README.md)

### [`PASS47`](PASS47.py) - Final Verification Suite
Comprehensive verification combining multiple analysis streams.

📄 [Detailed Documentation](PASS47_README.md)

## 📁 Repository Structure

```
pass_scripts/
├── README.md                 # This file
├── DOCUMENTATION.md          # Complete script catalog
├── PASS00.py - PASS48.py    # 49 verification scripts
├── PASS34_README.md         # Priority script docs
├── PASS36_README.md
├── PASS42_README.md
├── PASS45_README.md
└── PASS47_README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pmir-verification.git
cd pmir-verification

# Install dependencies
pip install numpy pandas matplotlib scipy
```

### Running a PASS Script

```bash
# Example: Run PASS34 (Growth Exponent Regression)
python PASS34.py \
  --in_csv data/pmir_results.csv \
  --outdir results/pass34 \
  --topology rr \
  --seed 42
```

## 📊 Data Format

Input CSV files should contain columns for:

- `topology`: Network type (e.g., 'rr', 'grid2d_periodic')
- `N`: Network size
- `lambda` or `lambda2`: Characteristic dimension measurements
- `seed`: Random seed for reproducibility
- Additional parameters specific to each test

## 📚 Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)**: Complete catalog of all 49 scripts
- **Individual READMEs**: Detailed documentation for priority scripts (PASS34, 36, 42, 45, 47)
- **Inline comments**: All scripts include extensive code documentation

## 🔬 Script Categories

Scripts are organized by function:

| Category | Count | Description |
|----------|-------|-------------|
| Lambda Analysis | 13 | Characteristic dimension calculations |
| Regression | 18 | Statistical modeling and fitting |
| Topology Analysis | 5 | Network structure effects |
| Bootstrap/Resampling | 5 | Uncertainty quantification |
| Time-Lock | 3 | Temporal dynamics |
| Other | 5 | Supporting utilities |

## 🤝 Contributing

This repository contains the exact verification tests used in the PMIR manuscript. For reproducibility, the scripts are preserved as-is. If you find issues or have questions, please open an issue.

## 📝 Citation

If you use these verification tests in your research, please cite:

```bibtex
@article{pmir2025,
  title={PMIR: Permutation-Mean Intrinsic Rotation Framework},
  author={[Your Name]},
  journal={[Journal]},
  year={2025}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [PMIR Manuscript](link-to-paper)
- [Full Dataset](link-to-data)
- [Analysis Code](link-to-analysis-repo)
