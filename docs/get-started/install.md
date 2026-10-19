---
description: Install trustcv with pip, add optional deep-learning frameworks, or install from source.
---

# Install

trustcv needs Python 3.10, 3.11 or 3.12.

```bash
pip install trustcv
```

This installs the splitters, checks, metrics and reporting, with scikit-learn as the only modelling dependency. Check the install:

```bash
trustcv --version
```

## Optional frameworks

Quote the brackets so shells such as zsh don't expand them.

| Command | Adds |
| --- | --- |
| `pip install "trustcv[pytorch]"` | PyTorch runner (`TorchCVRunner`) |
| `pip install "trustcv[tensorflow]"` | Keras runner (`KerasCVRunner`) and scikit-learn wrappers |
| `pip install "trustcv[monai]"` | MONAI runner for medical imaging |
| `pip install "trustcv[jax]"` | JAX/Flax runner |
| `pip install "trustcv[multilabel]"` | Iterative stratification for multilabel splitters |
| `pip install "trustcv[all]"` | All of the above, plus development tools |

## From source

```bash
git clone https://github.com/ki-smile/trustcv.git
cd trustcv
pip install -e ".[dev]"
```

Next: the [quickstart](quickstart.md).
