---
description: The trustcv command line tool and its seven runnable example recipes.
---

# Command line and examples

Installing trustcv adds a `trustcv` command.

```bash
trustcv --help               # quick start
trustcv about                # version, links, citation
trustcv examples             # list the 7 numbered recipes
trustcv examples 1 --run     # run recipe 1
trustcv examples 3 --copy .  # copy recipe 3 into the current folder to edit
```

| # | Recipe | Shows |
| --- | --- | --- |
| 1 | `all_cv_methods_demo` | Every splitter on one dataset |
| 2 | `data_leakage_detection_demo` | Patient, temporal and feature leakage being detected |
| 3 | `heart_disease_classification` | Patient-grouped CV with leakage checks |
| 4 | `heart_disease_prediction` | Comparing CV strategies |
| 5 | `multisite_clinical_trial` | Grouped CV for patients nested in hospitals |
| 6 | `icu_patient_monitoring` | Temporal CV for ICU time series |
| 7 | `framework_agnostic_demo` | One runner across scikit-learn, PyTorch and TensorFlow (deep-learning parts need those frameworks) |

Heavier examples for PyTorch, TensorFlow, JAX and MONAI are in the [`examples/`](https://github.com/ki-smile/trustcv/tree/main/examples) folder of the repository.
