---
description: How to record cross-validation choices, integrity checks, confidence intervals and subgroup results with trustcv for clinical and regulatory review.
---

# Documenting validation

<div class="tcv-disclaimer" markdown>trustcv supports documentation of validation choices. It does not make a model or device compliant with FDA, EU MDR or any other regulation.</div>

This page shows how to produce the parts of a validation record that come from cross-validation: how the data were split, which integrity checks ran and what they found, how uncertainty was estimated, and how subgroups performed. It does not cover the rest of a technical file: intended use, risk management, the reference standard, clinical performance studies or post-market monitoring.

## What reviewers commonly look for

The statements below summarise public documents. They are a starting point, not a list of requirements; read the source and get regulatory advice for your device and market.

| Topic | What the source says | Source |
| --- | --- | --- |
| Independent test data | Training and test data should be selected and maintained to be appropriately independent, considering sources of dependence such as the patient, data acquisition and site. (Paraphrase of guiding principle 4.) | FDA, Health Canada and MHRA, *Good Machine Learning Practice for Medical Device Development: Guiding Principles* (October 2021); extended by IMDRF in January 2025 |
| Representative data | Data sets should represent the intended patient population, for example in age, sex, race and ethnicity. (Paraphrase of guiding principle 3.) | Same |
| Subgroup performance | The TRIPOD+AI reporting guideline asks authors to evaluate model performance in key subgroups (for example, sociodemographic ones), and refers studies with clustered data to TRIPOD-Cluster. | Collins et al., *BMJ* 2024;385:e078378 |
| Clinical evaluation (EU) | Article 61 of the MDR requires clinical evaluation for devices of every class, not only higher-risk ones. | Regulation (EU) 2017/745, Art. 61 |
| Software lifecycle (EU) | Software is to be developed and manufactured according to the state of the art, taking into account the development life cycle, risk management, information security, verification and validation. | Regulation (EU) 2017/745, Annex I, Section 17.2 |
| Software as a device (EU) | Guidance on whether software qualifies as a medical device and how it is classified (Rule 11). | MDCG 2019-11 |
| Clinical evaluation of software (EU) | Guidance that structures evidence as valid clinical association, technical performance and clinical performance. | MDCG 2020-1 |

Cross-validation on retrospective data is evidence about technical performance on the data you had. On its own it does not establish clinical performance.

FDA guidance does not prescribe split ratios. Holding out 15–20% of patients as a locked test set, or using 5- or 10-fold cross-validation, is common practice. Whichever you choose, justify it from the data size and structure.

## Validate with patients kept together

When a patient contributes several rows (visits, images, samples), split by patient. Pass the patient IDs as `groups=` and use a group-aware method such as `stratified_group_kfold`. The example uses synthetic data: 120 patients with 3 visits each.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import trustcv
from trustcv import TrustCV

rng = np.random.default_rng(0)
n_patients, visits = 120, 3
patients = np.repeat(np.arange(n_patients), visits)
sex = np.repeat(rng.choice(["female", "male"], n_patients), visits)
risk = rng.normal(size=n_patients)
y = np.repeat((risk + rng.normal(0, 1, n_patients) > 0).astype(int), visits)
X = np.column_stack([risk[patients] + rng.normal(0, 0.5, patients.size),
                     rng.normal(size=(patients.size, 4))])

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
validator = TrustCV(method="stratified_group_kfold", n_splits=5, random_state=0,
                    ci_method="corrected_t")
result = validator.validate(model=model, X=X, y=y, groups=patients)

print(result.overall_status)
for name, check in result.checks.items():
    print(f"{name:26} {check.status:12} {check.message[:60]}")
```

Report `overall_status` together with every check, including those that are `NOT_CHECKED` or `INFO`. [Reading results](../get-started/reading-results.md) explains each status and what a `PASSED` status does not prove.

## Report intervals, not only means

Fold scores from K-fold cross-validation are correlated, because the training sets overlap. Percentiles or a plain t-interval of five fold scores give intervals that are too narrow. trustcv offers two alternatives:

| `ci_method` | What it does | When to use it |
| --- | --- | --- |
| `corrected_t` (default) | t-interval over fold scores with the Nadeau–Bengio variance correction for overlapping training sets. | Any splitter; the general default. |
| `oof_bootstrap` | Pools out-of-fold predictions and bootstraps them, resampling whole groups when `groups=` is given. | Splitters whose test folds cover each row exactly once (K-fold variants). Otherwise trustcv falls back to `corrected_t` with a warning. |

```python
print(result.ci_method, result.ci_level)
print("AUC", round(result.mean_scores["roc_auc"], 3),
      tuple(round(v, 3) for v in result.confidence_intervals["roc_auc"]))

boot = TrustCV(method="stratified_group_kfold", n_splits=5, random_state=0,
               ci_method="oof_bootstrap", n_bootstrap=500).validate(
    model=model, X=X, y=y, groups=patients)
print(boot.ci_method, tuple(round(v, 3) for v in boot.confidence_intervals["roc_auc"]))
```

State the method and level next to every interval, for example "AUC 0.74 (95% CI 0.61 to 0.87, corrected t-interval over 5 patient-grouped folds)". The [confidence intervals guide](../guides/confidence-intervals.md) covers the methods in more detail.

## Record what you ran

Most of a methods description can be read from the validator and the result, so it does not have to be retyped.

| Item | Where it comes from |
| --- | --- |
| Split method, number of folds, seed | `TrustCV(method=, n_splits=, random_state=)` |
| Grouping unit | What `groups=` contains (patient, site, scanner) |
| Rows per fold | `result.fold_details[i]["n_train"]`, `["n_val"]` |
| Per-fold scores | `result.scores` |
| Interval method and level | `result.ci_method`, `result.ci_level` |
| Integrity checks | `result.checks` (status, message, details) and `result.overall_status` |
| Suggested fixes | `result.recommendations` |
| Software version | `trustcv.__version__` |

```python
record = {
    "trustcv_version": trustcv.__version__,
    "method": validator.method,
    "n_splits": validator.n_splits,
    "random_state": validator.random_state,
    "grouping_unit": "patient",
    "n_patients": int(np.unique(patients).size),
    "n_rows": int(y.size),
    "fold_sizes": [(f["n_train"], f["n_val"]) for f in result.fold_details],
    "ci_method": result.ci_method,
    "overall_status": result.overall_status,
    "checks": {k: c.status for k, c in result.checks.items()},
}
print(record)
```

Things no check can record for you: whether preprocessing, feature selection or tuning happened before `validate()`, how the reference standard was established, and whether the analysis plan was fixed before the results were seen. Write these down yourself.

## Subgroup results

trustcv has no dedicated subgroup function. A practical approach is to collect out-of-fold predictions with the same patient-grouped splitter, then compute metrics per subgroup with `ClinicalMetrics.calculate_all`. Passing `groups=` makes its AUC interval resample whole patients.

```python
from sklearn.model_selection import cross_val_predict
from trustcv import ClinicalMetrics, StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, random_state=0)
proba = cross_val_predict(model, X, y, cv=cv, groups=patients, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = ClinicalMetrics()
for group in ["female", "male"]:
    m = sex == group
    res = metrics.calculate_all(y[m], pred[m], proba[m], groups=patients[m])
    lo, hi = res["auc_roc_ci"]
    print(f"{group:7} patients={np.unique(patients[m]).size:3} "
          f"AUC={res['auc_roc']:.3f} (95% CI {lo:.3f} to {hi:.3f})")
```

Points to keep in mind:

- Choose the subgroups before looking at the results, and report all of them, not only those that look good.
- Small subgroups give wide intervals. Report the number of patients in each.
- In `ClinicalMetrics`, `groups=` affects only the AUC interval. Intervals for sensitivity, specificity and the predictive values treat rows as independent, so with repeated rows per patient they are too narrow. Summarise to one row per patient first if you report them.
- A 0.5 threshold is used here for illustration. Report the threshold you used and how it was chosen.

## RegulatoryReport

`trustcv.reporting.RegulatoryReport` collects what is described above into one file. `generate_from_validator` reads the validator's last result and records:

- the split method and number of folds;
- per-fold scores of one metric (the first of `accuracy`, `roc_auc`, `f1` that is present);
- every integrity check and an overall status, recomputed from the checks so that incomplete evidence is never promoted to `PASSED`;
- dataset counts, if you call `add_dataset_info`.

```python
import json
from trustcv.reporting import RegulatoryReport

report = RegulatoryReport(
    model_name="Example model", model_version="0.1", manufacturer="Example lab",
    intended_use="Research use only; synthetic data.",
)
report.add_dataset_info(n_patients=n_patients, n_samples=y.size, n_features=X.shape[1])
path = report.generate_from_validator(validator, output_path="validation_record.json", format="json")

saved = json.load(open(path))
print(saved["validation"]["method"], saved["validation"]["overall_status"])
print(saved["validation"]["cv_results"]["scores"])
```

Prefer `format="json"` as the record. The HTML format in 1.2.1 has known limitations; edit its output before sharing it:

- it shows 0 for sensitivity, specificity, AUC and the confusion matrix unless `calculate_clinical_metrics` was called;
- it labels the summary score "Mean Accuracy" whichever metric was recorded, and shows no confidence interval for it;
- it contains a fixed "Regulatory Compliance" sentence stating that the report complies with the chosen standard, and an approval block. Remove both: neither trustcv nor this report can establish compliance.

To draft a summary from numbers you already have, see the [report draft tool](report-draft.md).

## Sources

- FDA, Health Canada, MHRA. [Good Machine Learning Practice for Medical Device Development: Guiding Principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles). October 2021.
- Collins GS, Moons KGM, Dhiman P, et al. [TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods](https://www.bmj.com/content/385/bmj-2023-078378). *BMJ* 2024;385:e078378.
- [Regulation (EU) 2017/745 on medical devices](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0745), Article 61 and Annex I.
- European Commission. [MDCG endorsed guidance documents](https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en), including MDCG 2019-11 and MDCG 2020-1.
- Nadeau C, Bengio Y. Inference for the generalization error. *Machine Learning* 2003;52:239–281.
