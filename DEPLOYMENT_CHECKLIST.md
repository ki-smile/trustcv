# 🚀 trustcv PyPI Deployment Checklist

## Pre-Deployment Verification

- [ ] **Version updated to 1.0.0** in:
  - [x] `trustcv/__init__.py`
  - [x] `setup.py`
  
- [ ] **Tests passing:**
  ```bash
  python test_all.py
  ```

- [ ] **Build test passing:**
  ```bash
  python test_build.py
  ```

- [ ] **No sensitive data:**
  ```bash
  grep -r "password\|secret\|key\|token\|api" trustcv/ --exclude-dir=__pycache__
  ```

- [ ] **Documentation ready:**
  - [ ] README.md has installation instructions
  - [ ] LICENSE file exists
  - [ ] AUTHORS.md is up to date

## Deployment Steps

### 1. Register on PyPI (if not done)
- [ ] Create account at https://pypi.org/account/register/
- [ ] Create account at https://test.pypi.org/account/register/
- [ ] Generate API tokens for both

### 2. Install Tools
```bash
pip install --upgrade pip setuptools wheel twine build
```

### 3. Build Package
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info/

# Build
python -m build
```

### 4. Test on TestPyPI
```bash
# Upload to test
python -m twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple trustcv
```

### 5. Deploy to PyPI
```bash
python -m twine upload dist/*
```

### 6. Verify
```bash
pip install trustcv
python -c "import trustcv; print(trustcv.__version__)"
```

### 7. Post-Deployment
- [ ] Create git tag: `git tag -a v1.0.0 -m "First release"`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Create GitHub release
- [ ] Update README with: `pip install trustcv`
- [ ] Announce release

## Quick Commands (Copy & Paste)

```bash
# Full deployment sequence
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine upload --repository testpypi dist/*
# Test, then:
python -m twine upload dist/*
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "name already taken" | Use different name or increment version |
| "invalid token" | Regenerate token at pypi.org |
| "no module named trustcv" | Check `packages=find_packages()` in setup.py |
| "403 Forbidden" | Check token permissions |

## Contact for Issues
- SMAILE Team: contact@smile.ki.se
- GitHub Issues: https://github.com/ki-smile/trustcv/issues