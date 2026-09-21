"""
verify_trustcv_fixes.py — run this yourself BEFORE and AFTER Codex's changes.

    cd trustcv
    pip install -e .
    python verify_trustcv_fixes.py

Each line prints ✅ (fixed) or ❌ (problem still present) plus the evidence,
so you can see the problem reproduced first and then confirm the fix.
"""

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from trustcv import TrustCV  # noqa: E402
from trustcv.checkers.leakage import DataLeakageChecker  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        try:
            ok, evidence = fn()
        except Exception as e:  # missing API = not fixed yet
            ok, evidence = False, f"{type(e).__name__}: {e}"
        RESULTS.append(ok)
        print(f"{'✅' if ok else '❌'}  {name}\n     → {evidence}\n")
        return fn
    return deco


def lr():
    return LogisticRegression(max_iter=2000)


def status(res, key):
    return res.checks[key].status


# 1 ------------------------------------------------------------------------
@check("1. Feature selection before CV on pure noise is NOT reported as 'PASSED'")
def _():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5000)); y = rng.integers(0, 2, 100)
    Xs = SelectKBest(k=20).fit_transform(X, y)
    res = TrustCV(method="stratified_kfold", random_state=0).validate(model=lr(), X=Xs, y=y)
    s = res.summary()
    auc = res.mean_scores.get("roc_auc", float("nan"))
    false_pass = "Leakage Check: PASSED" in s
    overall = getattr(res, "overall_status", "(no overall_status)")
    ok = (not false_pass) and overall != "PASSED"
    return ok, f"AUC on noise = {auc:.2f}; overall_status = {overall}; 'Leakage Check: PASSED' shown = {false_pass}"


# 2 ------------------------------------------------------------------------
@check("2. No groups given → patient/group leakage is NOT_CHECKED (not PASSED)")
def _():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(method="stratified_kfold").validate(model=lr(), X=X, y=y)
    st = status(res, "group_leakage")
    return st == "NOT_CHECKED", f"group_leakage = {st}"


# 3 ------------------------------------------------------------------------
@check("3. A crashing leakage detector is reported as ERROR (not silently PASSED)")
def _():
    class Crash:
        def check(self, *a, **k):
            raise RuntimeError("boom")
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV().validate(model=lr(), X=X, y=y, leakage_checker=Crash())
    legacy = res.leakage_check.get("has_leakage")
    st = status(res, "external_leakage_detector")
    return st == "ERROR", f"external_leakage_detector = {st} (legacy has_leakage = {legacy})"


# 4 ------------------------------------------------------------------------
@check("4. A feature that encodes the label is flagged")
def _():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300); X = rng.normal(size=(300, 10))
    X[:, 3] = y + rng.normal(scale=0.01, size=300)
    res = TrustCV().validate(model=lr(), X=X, y=y)
    st = status(res, "target_leakage_features")
    return st in {"WARNING", "FAILED"}, f"target_leakage_features = {st}, AUC = {res.mean_scores['roc_auc']:.2f}"


# 5 ------------------------------------------------------------------------
@check("5. Permutation check catches a broken split (test rows inside train)")
def _():
    class Overlap(BaseCrossValidator):
        def get_n_splits(self, X=None, y=None, groups=None):
            return 5
        def _iter_test_indices(self, X=None, y=None, groups=None):
            yield from np.array_split(np.arange(len(X)), 5)
        def split(self, X, y=None, groups=None):
            for te in self._iter_test_indices(X):
                yield np.arange(len(X)), te
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 5)); y = rng.integers(0, 2, 150)
    res = TrustCV(permutation_check=True, n_permutations=10).validate(
        model=KNeighborsClassifier(1), X=X, y=y, cv=Overlap())
    p = res.permutation
    return status(res, "permutation_sanity") == "FAILED", (
        f"observed = {p['observed']:.2f}, null mean on shuffled labels = {p['null_mean']:.2f} "
        f"(should be ~{p['chance_level']})")


# 6 ------------------------------------------------------------------------
@check("6. 95% CI for AUC contains the true value (0.5) on noise ≥ 85% of the time")
def _():
    hits, n = 0, 40
    for s in range(n):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(120, 50)); y = rng.integers(0, 2, 120)
        res = TrustCV(random_state=s, n_bootstrap=500).validate(model=lr(), X=X, y=y)
        lo, hi = res.confidence_intervals["roc_auc"]
        hits += lo <= 0.5 <= hi
    cov = hits / n
    return cov >= 0.85, f"coverage with default ci_method '{TrustCV().ci_method}' = {cov:.0%} (target ≈ 95%)"


# 7 ------------------------------------------------------------------------
@check("7. Class imbalance is information, not a FAILED check")
def _():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 5)); y = (rng.random(400) < 0.05).astype(int); X[y == 1] += 1
    res = TrustCV().validate(model=lr(), X=X, y=y)
    shown = "Class Balance: FAILED" in res.summary()
    return not shown, f"'Class Balance: FAILED' shown = {shown}"


# 8 ------------------------------------------------------------------------
@check("8. Covariate shift is reported as shift, not as leakage")
def _():
    rng = np.random.default_rng(0)
    rep = DataLeakageChecker(verbose=False).check_cv_splits(
        rng.normal(size=(200, 5)), rng.normal(loc=3, size=(200, 5)))
    ok = (not rep.has_leakage) and ("covariate_shift" in rep.details)
    return ok, f"has_leakage = {rep.has_leakage}, types = {rep.leakage_types}, detail keys = {list(rep.details)}"


# 9 ------------------------------------------------------------------------
@check("9. Two independent random samples are NOT flagged as near-duplicates")
def _():
    flagged = 0
    for s in range(5):
        rng = np.random.default_rng(s)
        rep = DataLeakageChecker(verbose=False).check_cv_splits(
            rng.normal(size=(200, 5)), rng.normal(size=(200, 5)))
        flagged += "near_duplicate" in rep.leakage_types
    return flagged == 0, f"false near-duplicate alarms in {flagged}/5 independent datasets"


# 10 -----------------------------------------------------------------------
@check("10. recommend_cv() gives a method, rationale and runnable splitter")
def _():
    from trustcv import recommend_cv
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(60), 4)
    y = np.repeat((rng.random(60) < 0.2).astype(int), 4)
    X = rng.normal(size=(240, 5))
    rec = recommend_cv(X, y, groups=groups)
    TrustCV().validate(model=lr(), X=X, y=y, groups=groups, cv=rec.splitter)
    return rec.category == "grouped", f"{rec.category} / {type(rec.splitter).__name__}: {rec.rationale[:90]}..."


# 11 -----------------------------------------------------------------------
@check("11. The correct workflow still works: Pipeline + independent samples → PASSED")
def _():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(declare_independent_samples=True).validate(
        model=make_pipeline(StandardScaler(), lr()), X=X, y=y)
    return res.overall_status == "PASSED", f"overall_status = {res.overall_status}, AUC = {res.mean_scores['roc_auc']:.3f}"


print("=" * 70)
print(f"{sum(RESULTS)}/{len(RESULTS)} checks fixed")
