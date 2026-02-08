# Contributing to PMIR Verification Tests

Thank you for your interest in the PMIR verification tests!

## Reproducibility Priority

This repository preserves the exact verification tests used in the PMIR manuscript.
For scientific reproducibility, the core scripts (PASS00-PASS48) are maintained as-is.

## How to Contribute

### Reporting Issues

If you find bugs or have questions:

1. Check existing issues first
2. Provide a minimal reproducible example
3. Include your Python version and dependencies
4. Describe expected vs. actual behavior

### Documentation Improvements

We welcome improvements to:
- README files
- Code comments
- Usage examples
- Tutorial notebooks

### Extensions

For new analyses or extensions:
- Create them in a separate directory (e.g., `extensions/`)
- Reference the original PASS scripts
- Clearly mark as extensions, not core tests

## Code Style

- Follow PEP 8
- Add docstrings to new functions
- Include type hints where helpful
- Comment complex logic

## Testing

Before submitting:
- Verify your code runs on Python 3.7+
- Test with sample data if available
- Check that existing tests still pass

## Questions?

Open an issue for discussion before starting major work.

Thank you for helping improve scientific reproducibility!
