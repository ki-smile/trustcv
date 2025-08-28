# Contributing to trustcv

Thank you for your interest in contributing to trustcv! This toolkit aims to improve medical machine learning validation practices, and we welcome contributions from the community.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 🚀 How to Contribute

### 1. Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**To report a bug:**
1. Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template
2. Include:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Minimal code example

### 2. Suggesting Features

We welcome feature suggestions that align with trustcv's mission.

**To suggest a feature:**
1. Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template
2. Include:
   - Clear use case
   - Medical/clinical context
   - Proposed implementation (if applicable)
   - Alternative solutions considered

### 3. Contributing Code

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/trustcv.git
cd trustcv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add/update tests
   - Update documentation

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   flake8 trustcv/
   black --check trustcv/
   
   # Run type checking
   mypy trustcv/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new cross-validation method for spatial data"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Code style changes
   - `perf:` Performance improvements

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a PR on GitHub.

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with these additions:
- Line length: 88 characters (Black default)
- Use type hints for function parameters and returns
- Document all public functions with docstrings (NumPy style)

### Example Code Style

```python
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_clinical_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "sensitivity",
    confidence_level: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate clinical metric with confidence interval.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    metric : str, default='sensitivity'
        Metric to calculate
    confidence_level : float, default=0.95
        Confidence level for interval
        
    Returns
    -------
    metric_value : float
        Calculated metric value
    confidence_interval : Tuple[float, float]
        Lower and upper bounds of CI
        
    Examples
    --------
    >>> sensitivity, (ci_lower, ci_upper) = calculate_clinical_metric(
    ...     y_true, y_pred, metric="sensitivity"
    ... )
    """
    # Implementation
    pass
```

### Testing Requirements

- All new features must have tests
- Maintain test coverage above 80%
- Use pytest for testing
- Include both unit and integration tests

### Documentation

- Update docstrings for all changes
- Update relevant .md files in `/docs`
- Add examples to docstrings
- Update notebooks if applicable

## 🏥 Medical/Clinical Considerations

When contributing medical-specific features:

1. **Cite sources**: Include references to medical literature
2. **Clinical validation**: Explain clinical relevance
3. **Safety checks**: Consider patient safety implications
4. **Regulatory aspects**: Note any FDA/CE considerations
5. **Privacy**: Ensure no PHI in examples

## 📚 Areas We Need Help

### High Priority
- [ ] Additional CV methods from literature
- [ ] Integration with medical imaging libraries
- [ ] Real-world clinical case studies
- [ ] Performance optimization
- [ ] More comprehensive tests

### Documentation
- [ ] Video tutorials
- [ ] Clinical use case examples
- [ ] Translation to other languages
- [ ] API documentation improvements

### Research
- [ ] Implementing new CV methods from recent papers
- [ ] Benchmarking against other libraries
- [ ] Clinical validation studies

## 🔄 Pull Request Process

1. **PR Title**: Use conventional commit format
2. **Description**: Use the PR template
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update if needed
5. **Review**: Address reviewer feedback
6. **Squash**: We squash commits on merge

### PR Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] No sensitive medical data included

## 📦 Release Process

We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

Releases are automated via GitHub Actions when tags are pushed.

## 💬 Getting Help

- **Discord**: [Join our community](https://discord.gg/medicalcv)
- **Discussions**: Use GitHub Discussions for questions
- **Email**: medicalcv@example.com

## 🙏 Recognition

Contributors are recognized in:
- [AUTHORS.md](AUTHORS.md) file
- Release notes
- Project documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make medical machine learning more reliable! 🏥🤖