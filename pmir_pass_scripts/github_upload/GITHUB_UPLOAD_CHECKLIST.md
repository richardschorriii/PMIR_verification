# GitHub Upload Checklist

## Pre-Upload Verification

- [ ] All 49 PASS scripts extracted (PASS00-PASS48)
- [ ] Priority scripts verified (PASS34, 36, 42, 45, 47)
- [ ] All scripts have proper headers
- [ ] README.md is complete and accurate
- [ ] DOCUMENTATION.md catalogs all scripts
- [ ] COMPLETE_INDEX.md provides detailed reference
- [ ] Individual READMEs for priority scripts
- [ ] LICENSE file present
- [ ] .gitignore configured
- [ ] CONTRIBUTING.md included

## Repository Setup

```bash
# 1. Create repository on GitHub
# Repository name: pmir-verification-tests
# Description: 49 verification tests for PMIR framework

# 2. Initialize local repository
cd github_upload
git init
git add .
git commit -m "Initial commit: 49 PASS verification scripts"

# 3. Connect to GitHub
git remote add origin https://github.com/yourusername/pmir-verification-tests.git
git branch -M main
git push -u origin main
```

## Post-Upload Tasks

- [ ] Verify all files uploaded correctly
- [ ] Check that READMEs render properly
- [ ] Add repository description
- [ ] Add topics/tags: `python`, `network-analysis`, `verification`, `pmir`
- [ ] Enable issues
- [ ] Add repository URL to manuscript
- [ ] Test clone and run on fresh system

## File Count Verification

Expected file counts:
- Python scripts: 49 (PASS00.py - PASS48.py)
- Documentation files: 8+ (README, DOCUMENTATION, INDEX, priority READMEs)
- Configuration files: 3 (LICENSE, .gitignore, CONTRIBUTING.md)

## Script Verification

Test that critical scripts run:

```bash
# Check PASS34
python scripts/PASS34.py --help

# Check PASS36
python scripts/PASS36.py --help

# Check PASS45
python scripts/PASS45.py --help
```

## Citation Check

- [ ] Update citation in README with actual paper details
- [ ] Add DOI if available
- [ ] Link to dataset if published

