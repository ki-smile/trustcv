#  Setup Instructions for trustcv

## Prerequisites

- Python 3.8+ 
- Anaconda or Miniconda installed
- Git

## 🔧 Installation Options

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/ki-smile/trustcv.git
cd trustcv

# Create conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate trustcv

# Install the package in development mode
pip install -e .

# Verify installation
python -c "import trustcv; print('trustcv installed successfully!')"
```

### Option 2: Using pip with virtual environment

```bash
# Clone the repository
git clone https://github.com/ki-smile/trustcv.git
cd trustcv

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development requirements (optional)
pip install -r requirements-dev.txt

# Install the package
pip install -e .
```

### Option 3: Quick Install (for users)

```bash
# Direct installation from GitHub
pip install git+https://github.com/ki-smile/trustcv.git

# Or with conda
conda create -n trustcv python=3.10
conda activate trustcv
pip install git+https://github.com/ki-smile/trustcv.git
```

## 📋 Verify Your Setup

### 1. Test Basic Import

```python
# Test basic functionality
import trustcv
from trustcv.splitters.iid import KFoldMedical
from trustcv.splitters.temporal import PurgedKFoldCV
from trustcv.splitters.grouped import GroupKFoldMedical
from trustcv.splitters.spatial import SpatialBlockCV

print("✅ All modules imported successfully!")
```

### 2. Run Quick Test

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from trustcv.splitters.iid import StratifiedKFoldMedical

# Create sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Test cross-validation
cv = StratifiedKFoldMedical(n_splits=5)
model = RandomForestClassifier(n_estimators=10)

scores = []
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
    scores.append(score)

print(f"✅ Cross-validation working! Mean score: {np.mean(scores):.3f}")
```

### 3. Test Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab

# Navigate to notebooks/ folder and open any notebook
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=trustcv --cov-report=html

# Run specific test file
pytest tests/test_iid_methods.py -v

# Run integration tests
pytest tests/test_integration_examples.py -v
```

## 📚 Running Examples

### Example 1: Heart Disease Prediction
```bash
cd examples
python heart_disease_prediction.py
```

### Example 2: ICU Patient Monitoring
```bash
python icu_patient_monitoring.py
```

### Example 3: Multi-site Clinical Trial
```bash
python multisite_clinical_trial.py
```

### Example 4: Disease Spread Modeling
```bash
python disease_spread_modeling.py
```

## 🌐 Running the Interactive Website

```bash
# Navigate to website directory
cd website

# Start a local server
# Option 1: Using Python
python -m http.server 8000

# Option 2: Using Node.js (if installed)
npx http-server

# Open browser and go to:
# http://localhost:8000
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Import Error: No module named 'trustcv'
```bash
# Make sure you're in the correct directory and run:
pip install -e .
```

#### 2. Missing dependencies
```bash
# Update all dependencies
pip install -r requirements.txt --upgrade
```

#### 3. Jupyter kernel not found
```bash
# Install kernel for the conda environment
python -m ipykernel install --user --name trustcv --display-name "Python (trustcv)"
```

#### 4. Permission denied errors
```bash
# On macOS/Linux, use sudo for global installations
sudo pip install -e .

# Or better, use --user flag
pip install --user -e .
```

#### 5. Conda environment activation issues
```bash
# Make sure conda is initialized
conda init bash  # or zsh, fish, etc.

# Restart terminal and try again
conda activate trustcv
```

## 📦 Package Structure

```
trustcv/
├── trustcv/              # Main package
│   ├── splitters/         # CV methods
│   │   ├── iid.py        # I.I.D. methods
│   │   ├── temporal.py   # Temporal methods
│   │   ├── grouped.py    # Grouped methods
│   │   └── spatial.py    # Spatial methods
│   ├── checkers/         # Validation tools
│   ├── metrics/          # Medical-specific metrics
│   └── visualization/    # Plotting utilities
├── examples/             # Example scripts
├── notebooks/            # Jupyter tutorials
├── tests/               # Test suite
├── website/             # Interactive website
├── docs/                # Documentation
├── environment.yml      # Conda environment
├── requirements.txt     # Pip requirements
├── requirements-dev.txt # Development requirements
└── setup.py            # Package setup
```

## 🆘 Getting Help

1. **Documentation**: Check the `docs/` folder
2. **Notebooks**: Interactive tutorials in `notebooks/`
3. **Website**: Interactive demos at http://localhost:8000
4. **Issues**: Report bugs at https://github.com/ki-smile/trustcv/issues
5. **Examples**: Working examples in `examples/`

## 🎯 Next Steps

1. **Explore Notebooks**: Start with `01_IID_Methods.ipynb`
2. **Try Interactive Tutorials**: Open `website/tutorials.html`
3. **Read Data Leakage Demo**: `10_Data_Leakage_Consequences.ipynb`
4. **Run Examples**: Try all 4 example scripts
5. **Test Your Data**: Apply appropriate CV methods to your medical data

## ✅ Checklist for Complete Setup

- [ ] Conda/virtual environment created
- [ ] All dependencies installed
- [ ] Package installed with `pip install -e .`
- [ ] Basic import test passed
- [ ] Quick test script works
- [ ] At least one example runs successfully
- [ ] Jupyter notebooks open correctly
- [ ] Tests run without errors
- [ ] Website loads in browser

## 📝 Notes

- Use Python 3.8+ for best compatibility
- GPU is not required but beneficial for large datasets
- Recommended RAM: 8GB+ for running all examples
- Internet connection needed for initial setup only

---

**Happy Cross-Validating!** 🏥✨

For questions or issues, please visit: https://github.com/ki-smile/trustcv/issues