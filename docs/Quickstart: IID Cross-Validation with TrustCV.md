# Quickstart: IID Cross-Validation with TrustCV

This tutorial shows how to:

1. Run a baseline IID CV with scikit-learn.
2. Switch to TrustCV’s IID splitters and `UniversalCVRunner`.
3. Use `TrustCVValidator` for a higher-level workflow.
4. Run leakage and balance checks.
5. Compute clinical metrics and generate a simple report.

**File:** `docs/quickstart_iid.md`  
(and a matching notebook: `notebooks/Quickstart_IID_TrustCV.ipynb`)


## 1. Load a standard dataset (e.g. breast cancer):

  ```python
  from sklearn.datasets import load_breast_cancer
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestClassifier
  
  X, y = load_breast_cancer(return_X_y=True)
  model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
  ```

------

## 2. Baseline: plain scikit-learn CV

- Show a short baseline using scikit-learn’s `StratifiedKFold` and a manual loop
   or `cross_val_score`, just to anchor the user:

  ```python
  from sklearn.model_selection import StratifiedKFold
  from sklearn.metrics import accuracy_score, roc_auc_score
  
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  
  accs, aucs = [], []
  for train_idx, test_idx in cv.split(X, y):
      model.fit(X[train_idx], y[train_idx])
      proba = model.predict_proba(X[test_idx])[:, 1]
      pred = (proba >= 0.5).astype(int)
      accs.append(accuracy_score(y[test_idx], pred))
      aucs.append(roc_auc_score(y[test_idx], proba))
  
  print(f"Accuracy: {sum(accs)/len(accs):.3f}, ROC AUC: {sum(aucs)/len(aucs):.3f}")
  ```

- Brief commentary: works, but you had to write the loop yourself and no leakage/balance checks.

------

## 3. Using TrustCV splitters + UniversalCVRunner

- Replace the scikit-learn splitter with `StratifiedKFold` and the manual loop
   with `UniversalCVRunner`:

  ```python
  from trustcv import StratifiedKFold, UniversalCVRunner
  
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
  
  results = runner.run(model=model, data=(X, y), metrics=["accuracy", "roc_auc"])
  print(results.summary())
  ```

- Explain what the summary shows:

  - Mean ± std per metric.
  - Same semantics as the manual loop, but less boilerplate.

------

## 4. Higher-level flow with TrustCVValidator

- Introduce `TrustCVValidator` for users who prefer a single entry point:

  ```python
  from trustcv import TrustCVValidator
  
  validator = TrustCVValidator(
      method="stratified_kfold",
      n_splits=5,
      metrics=["accuracy", "roc_auc", "f1"],
      check_leakage=True,
      check_balance=True,
      return_confidence_intervals=True,
  )
  
  val_result = validator.validate(model=model, X=X, y=y)
  print(val_result.summary())
  ```

- Explain:

  - `method="stratified_kfold"` selects the IID stratified splitter.
  - `check_leakage=True`, `check_balance=True` trigger internal analyses.
  - The summary includes mean, std, and 95% confidence intervals.

------

## 5. Leakage and class-balance checks

- Show explicit use of `DataLeakageChecker` and `BalanceChecker`:

  ```python
  from trustcv import DataLeakageChecker, BalanceChecker
  
  leak_report = DataLeakageChecker().check(X=X, y=y, n_splits=5, random_state=42)
  balance_report = BalanceChecker().check_class_balance(y)
  
  print(leak_report.summary)
  print(balance_report)
  ```

- Briefly interpret:

  - No leakage detected vs. warnings.
  - Imbalance ratios and suggestions (e.g., “use stratification”).
