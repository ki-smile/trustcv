---
title: trustcv
description: Trustworthy cross-validation for medical and regulated machine learning. Leakage-aware splitters, integrity checks with an honest overall status, and corrected confidence intervals.
hide:
  - navigation
  - toc
---

<div class="tcv-hero" markdown>

# Cross-validation you can defend

<p class="tcv-lede">trustcv chooses a cross-validation split that matches how your data was collected, checks the folds you actually scored for leakage, and reports what it could and could not verify. Built for medical and regulated machine learning; works with scikit-learn, PyTorch, TensorFlow, MONAI and JAX.</p>

```bash
pip install trustcv
```

[Get started](get-started/quickstart.md){ .md-button .md-button--primary }
[See how splits work](learn/visualisations.md){ .md-button }

</div>

<div class="tcv-cards" markdown>

<div class="tcv-card" markdown>
### :material-source-branch: The right split
32 methods (the 29 in the paper plus later additions) for independent, grouped, temporal, spatial and multilabel data. `recommend_cv` picks one from how your data is structured.
</div>

<div class="tcv-card" markdown>
### :material-shield-search: Leakage checks
Group overlap, duplicates, target proxies, clustered rows and identifier columns, checked on the folds that were scored.
</div>

<div class="tcv-card" markdown>
### :material-chart-bell-curve: Honest intervals
Confidence intervals that account for overlapping training sets (`corrected_t`) or resample whole patients (`oof_bootstrap`).
</div>

<div class="tcv-card" markdown>
### :material-file-document-check: A status you can read
Every result ends in one overall status, and each check says what it established and what it could not see.
</div>

</div>

## A first validation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV, recommend_cv

X, y = load_breast_cancer(return_X_y=True)
plan = recommend_cv(X, y)  # independent rows here, so stratified K-fold

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
result = TrustCV(method=plan.method, declare_independent_samples=True).validate(
    model=model, X=X, y=y, cv=plan.splitter
)
print(result.overall_status)  # PASSED
print(result.summary())
```

Preprocessing sits inside the `Pipeline`, so it is refit in every fold. Rows that share a patient need `groups=`: see the [quickstart](get-started/quickstart.md).

## What the result tells you

| Overall status | Meaning |
| --- | --- |
| <span class="tcv-status passed">PASSED</span> | Every applicable check passed. Not proof that no leakage exists: some kinds cannot be seen from the data. |
| <span class="tcv-status review">NEEDS_REVIEW</span> | Signs of possible leakage that cannot be proven automatically. A person should confirm them. |
| <span class="tcv-status unverified">NOT_FULLY_VERIFIED</span> | Nothing suspicious, but something could not be checked (for example, no patient IDs were given). |
| <span class="tcv-status failed">FAILED</span> | A check found leakage, such as the same patient in training and test. |

[How to read every check](get-started/reading-results.md)

## Learn more

<div class="tcv-cards" markdown>

<div class="tcv-card" markdown>
### [Choosing a split](guides/choosing-cv.md)
Match the split to repeated patients, time order, sites and geography.
</div>

<div class="tcv-card" markdown>
### [Data leakage](guides/data-leakage.md)
What each check catches, and the leaks no check can see.
</div>

<div class="tcv-card" markdown>
### [Deep learning](guides/deep-learning.md)
Runners for PyTorch, TensorFlow, MONAI and JAX, and early stopping without touching the test fold.
</div>

<div class="tcv-card" markdown>
### [Documenting validation](regulatory/documentation-guide.md)
Record split choices and uncertainty for clinical and regulatory review.
</div>

</div>

## Podcast

<div class="tcv-card" markdown>
**The reproducibility problem in medical AI** · 21:50 · AI-generated audio about data leakage and validation in clinical settings.

<audio controls preload="none" src="https://github.com/ki-smile/trustcv/raw/main/website/trustcv.mp3" style="width: 100%"></audio>
</div>

<small>Developed at [SMAILE](https://smile.ki.se), Karolinska Institutet's core facility for AI. trustcv supports documentation of validation choices; it does not make a model or device compliant with any regulation.</small>
