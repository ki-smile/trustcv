from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trustcv",
    version="1.0.5",
    author="SMAILE (Stockholm Medical AI and Learning Environments), Karolinska Institutet",
    description="Trustworthy Cross-Validation: Framework-agnostic CV with data leakage detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ki-smile/trustcv",
    project_urls={
        "SMAILE Lab": "https://smile.ki.se",
        "Documentation": "https://ki-smile.github.io/trustcv",
        "Bug Tracker": "https://github.com/ki-smile/trustcv/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "plotly>=5.0.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        # Framework-specific installations
        "pytorch": [
            "torch>=1.10.0",
            "torchvision>=0.11.0",
        ],
        "tensorflow": [
            "tensorflow>=2.10.0",
        ],
        "monai": [
            "monai>=1.0.0",
            "nibabel>=3.2.0",  # For medical image I/O
            "torch>=1.10.0",  # MONAI requires PyTorch
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "flax>=0.6.0",
        ],
        "xgboost": [
            "xgboost>=1.7.0",
        ],
        "lightgbm": [
            "lightgbm>=3.3.0",
        ],
        "catboost": [
            "catboost>=1.1.0",
        ],
        # Complete installation
        "all": [
            "torch>=1.10.0",
            "torchvision>=0.11.0",
            "tensorflow>=2.10.0",
            "monai>=1.0.0",
            "nibabel>=3.2.0",
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "optuna>=3.0.0",  # For hyperparameter tuning
        ],
        # Development dependencies
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "isort>=5.10.0",
        ],
        # Documentation dependencies
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.18.0",
        ],
    },
    keywords=[
        "machine learning",
        "cross-validation", 
        "AI",
        "healthcare",
        "clinical research",
        "regulatory compliance",
        "FDA",
        "CE MDR",
        "pytorch",
        "tensorflow",
        "MONAI",
        "framework-agnostic"
    ],
)