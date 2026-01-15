"""
Trustworthy Cross-Validation Validators

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Main validation classes with medical-specific features
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.utils.validation import check_array


@dataclass
class ValidationResult:
    """Results from medical cross-validation"""

    scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    fold_details: List[Dict]
    leakage_check: Dict[str, bool]
    recommendations: List[str]
    ci_method: str = ""
    ci_level: float = 0.95

    def summary(self) -> str:
        """Generate summary report"""
        summary = "=== Trustworthy Cross-Validation Results ===\n\n"
        ci_tag = ""
        if self.ci_method:
            ci_tag = f" (method: {self.ci_method})"
        ci_percent = self.ci_level * 100 if self.ci_level else 95.0
        ci_percent_str = f"{ci_percent:.1f}".rstrip("0").rstrip(".")
        summary += f"Performance Metrics (mean +/- std){ci_tag}:\n"
        # Avoid duplicate lines when both 'metric' and 'test_metric' keys exist
        seen = set()
        for metric, mean_val in self.mean_scores.items():
            display_metric = metric.replace("test_", "")
            if display_metric in seen:
                continue
            std_val = self.std_scores.get(metric, self.std_scores.get(display_metric, 0.0))
            ci_lower, ci_upper = self.confidence_intervals.get(display_metric, (0.0, 0.0))
            ci_label = self.ci_method
            if ci_label:
                ci_label = f" ({ci_label})"
            if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                summary += f"  {display_metric}: {mean_val:.3f} +/- {std_val:.3f} "
                summary += f"[{ci_percent_str}% CI{ci_label}: {ci_lower:.3f}-{ci_upper:.3f}]\n"
            else:
                summary += f"  {display_metric}: {mean_val:.3f} +/- {std_val:.3f} "
                summary += f"[{ci_percent_str}% CI: n/a]\n"
            seen.add(display_metric)

        summary += "\nData Integrity Checks:\n"
        friendly_names = {
            "no_duplicate_samples": "Duplicate Samples",
            "no_patient_leakage": "Patient Leakage Separation",
            "has_leakage": "External Leakage Detector",
            "balanced_classes": "Class Balance",
        }
        handled = set()
        leakage_keys = [
            k
            for k in ("no_duplicate_samples", "no_patient_leakage", "has_leakage")
            if k in self.leakage_check
        ]
        if leakage_keys:
            leakage_status = all(self.leakage_check[k] for k in leakage_keys)
            summary += f"  Leakage Check: {'PASSED' if leakage_status else 'FAILED'}\n"
            handled.update(leakage_keys)
        if "balanced_classes" in self.leakage_check:
            balanced = self.leakage_check["balanced_classes"]
            summary += f"  Class Balance: {'PASSED' if balanced else 'FAILED'}\n"
            handled.add("balanced_classes")
        for check, passed in self.leakage_check.items():
            if check in handled:
                continue
            label = friendly_names.get(check, check.replace("_", " ").title())
            status = "PASSED" if passed else "FAILED"
            summary += f"  {label}: {status}\n"

        if self.recommendations:
            summary += "\nRecommendations:\n"
            for rec in self.recommendations:
                summary += f"  - {rec}\n"

        return summary

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            "mean_scores": self.mean_scores,
            "std_scores": self.std_scores,
            "confidence_intervals": self.confidence_intervals,
            "ci_method": self.ci_method,
            "ci_level": self.ci_level,
            "leakage_check": self.leakage_check,
            "recommendations": self.recommendations,
        }


class TrustCVValidator:
    """
    Main validator class for medical machine learning

    Features:
    - Automatic patient-level splitting
    - Data leakage detection
    - Clinical metrics calculation
    - Regulatory compliance reporting
    """

    def __init__(
        self,
        method: str = "stratified_kfold",
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
        check_leakage: bool = True,
        check_balance: bool = True,
        compliance: Optional[str] = None,
        *,
        metrics: Optional[List[str]] = None,
        return_confidence_intervals: bool = True,
        ci_level: float = 0.95,
        ci_method: str = "bootstrap",
        n_bootstrap: int = 1000,
        holdout_test_size: Union[float, int] = 0.2,
        holdout_stratify: bool = False,
        repeated_kfold_repeats: int = 1,
        repeated_kfold_stratify: bool = False,
        n_repeats: Optional[int] = None,
        lpocv_p: int = 2,
        p: Optional[int] = None,
        monte_carlo_iterations: int = 50,
        n_iterations: Optional[int] = None,
        iterations: Optional[int] = None,
        monte_carlo_test_size: Union[float, int] = 0.2,
        mc_test_size: Optional[Union[float, int]] = None,
        bootstrap_validation_iterations: int = 200,
        bootstrap_iterations: Optional[int] = None,
        bootstrap_validation_estimator: str = "standard",
        bootstrap_estimator: Optional[str] = None,
        test_size: Optional[Union[float, int]] = None,
        stratify: Optional[bool] = None,
    ):
        """
        Initialize TrustCV Validator

        Parameters:
        -----------
        method : str
            Cross-validation method ('kfold', 'stratified_kfold',
            'patient_grouped_kfold', 'temporal', 'holdout',
            'repeated_kfold', 'loocv', 'lpocv', 'monte_carlo', 'bootstrap')
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed for reproducibility
        shuffle : bool
            Whether to shuffle data when using k-fold style splitters (default: True)
        check_leakage : bool
            Whether to check for data leakage
        check_balance : bool
            Whether to check class balance
        compliance : str
            Regulatory compliance mode ('FDA', 'CE', None)
        holdout_test_size : float or int
            Fraction or absolute count for hold-out validation when ``method='holdout'``
        holdout_stratify : bool
            If True, enables stratified hold-out splitting (uses ``y`` labels)
        repeated_kfold_repeats : int
            Number of repetitions when ``method='repeated_kfold'`` (alias: ``n_repeats``)
        repeated_kfold_stratify : bool
            Whether to use stratified repeats (alias: ``stratify`` when ``method='repeated_kfold'``)
        lpocv_p : int
            Number of samples to leave out for ``method='lpocv'`` (alias: ``p``)
        monte_carlo_iterations : int
            Random split iterations for ``method='monte_carlo'`` (aliases: ``n_iterations``, ``iterations``)
        monte_carlo_test_size : float or int
            Fraction or count for Monte Carlo test splits (alias: ``mc_test_size``; ``test_size`` also handled)
        bootstrap_validation_iterations : int
            Number of bootstrap resamples when ``method='bootstrap'`` (alias: ``bootstrap_iterations``)
        bootstrap_validation_estimator : str
            Bootstrap estimator variant ('standard', '.632', '.632+') (alias: ``bootstrap_estimator``)
        test_size : float or int, optional
            Alias for ``holdout_test_size`` (and, when using Monte Carlo splits, ``monte_carlo_test_size``)
        stratify : bool, optional
            Alias for ``holdout_stratify`` to toggle stratified hold-out splitting
        """
        # Accept multiple naming styles (canonical and sklearn-style)
        self.method = self._normalize_method(method)
        method_key = self.method
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = bool(shuffle)
        self.check_leakage = check_leakage
        self.check_balance = check_balance
        self.compliance = compliance
        self.metrics = self._normalize_metric_list(metrics)
        self.return_confidence_intervals = bool(return_confidence_intervals)
        self.ci_level = float(ci_level)
        if not 0 < self.ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1 (exclusive).")
        self.ci_method = ci_method
        self.n_bootstrap = int(n_bootstrap)

        holdout_size_val = holdout_test_size
        if test_size is not None:
            holdout_size_val = test_size
        self.holdout_test_size = self._validate_split_size(holdout_size_val, "holdout_test_size")
        method_key = self.method
        self.holdout_stratify = bool(holdout_stratify)
        self.repeated_kfold_stratify = bool(repeated_kfold_stratify)
        if stratify is not None:
            if method_key == "holdout":
                self.holdout_stratify = bool(stratify)
            elif method_key == "repeated_kfold":
                self.repeated_kfold_stratify = bool(stratify)

        repeats_val = n_repeats if n_repeats is not None else repeated_kfold_repeats
        self.repeated_kfold_repeats = max(int(repeats_val), 1)

        lp_val = p if p is not None else lpocv_p
        self.lpocv_p = max(int(lp_val), 1)

        mc_iter_val = monte_carlo_iterations
        if n_iterations is not None:
            mc_iter_val = n_iterations
        elif iterations is not None:
            mc_iter_val = iterations
        self.monte_carlo_iterations = max(int(mc_iter_val), 1)

        mc_size_val = monte_carlo_test_size
        if mc_test_size is not None:
            mc_size_val = mc_test_size
        elif test_size is not None and method_key == "monte_carlo":
            mc_size_val = test_size
        self.monte_carlo_test_size = self._validate_split_size(mc_size_val, "monte_carlo_test_size")

        boot_iter_val = (
            bootstrap_iterations
            if bootstrap_iterations is not None
            else bootstrap_validation_iterations
        )
        self.bootstrap_validation_iterations = max(int(boot_iter_val), 1)
        self.bootstrap_validation_estimator = (
            str(bootstrap_estimator)
            if bootstrap_estimator is not None
            else bootstrap_validation_estimator
        )

        self._cv_splitter = None
        self._setup_splitter()

    # --- public API ---
    def validate(
        self,
        *,
        model,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        patient_ids: Optional[Union[np.ndarray, "pd.Series"]] = None,
        groups: Optional[Union[np.ndarray, "pd.Series"]] = None,
        cv: Optional[BaseCrossValidator] = None,
        leakage_checker: Optional[Any] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ValidationResult":
        """
        Run cross-validation with the requested metrics and return a ValidationResult.

        Parameters
        ----------
        model : estimator implementing fit/predict
        X, y : array-like
        patient_ids : array-like, optional
            Alias for groups; convenient when using patient-level splitters
        groups : array-like, optional
            Explicit group labels for the cross-validator.
        cv : cross-validator, optional
            If None, uses the validator's configured splitter
        leakage_checker : DataLeakageChecker, optional
            If provided, runs leakage checks on the dataset
        sample_weight : array-like, optional
            Per-sample weights passed to estimator.fit when supported
        """
        import numpy as _np
        from sklearn.base import clone as _sk_clone

        X_arr = _np.asarray(X) if not hasattr(X, "iloc") else X
        y_arr = _np.asarray(y) if not hasattr(y, "iloc") else y
        n = len(X_arr)
        group_labels = groups
        if group_labels is None and patient_ids is not None:
            group_labels = patient_ids
        if group_labels is not None:
            group_len = len(group_labels)
            if group_len != n:
                raise ValueError(
                    f"groups/patient_ids length mismatch: expected {n}, got {group_len}"
                )

        # Track binary class labels so we can compute sensitivity/specificity when requested
        y_for_class = y_arr.to_numpy() if hasattr(y_arr, "to_numpy") else _np.asarray(y_arr)
        binary_labels: Optional[Tuple[Any, Any]] = None
        if y_for_class.ndim == 1:
            unique_labels = _np.unique(y_for_class)
            if unique_labels.size == 2:
                # confusion_matrix orders labels ascending; keep the same ordering here
                binary_labels = (unique_labels[0], unique_labels[1])

        splitter = cv if cv is not None else self._cv_splitter
        if splitter is None:
            # default safe fallback
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )

        # storage
        metric_list = list(self.metrics)
        per_metric_scores: Dict[str, List[float]] = {m: [] for m in metric_list}
        fold_details: List[Dict[str, Any]] = []

        # iterate folds
        # Most trustcv/sklearn splitters accept (X, y, groups)
        split_groups = group_labels
        for k, (tr, te) in enumerate(splitter.split(X_arr, y_arr, split_groups), 1):
            # train/val slices
            if hasattr(X_arr, "iloc"):
                X_tr, X_te = X_arr.iloc[tr], X_arr.iloc[te]
            else:
                X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr = y_arr.iloc[tr] if hasattr(y_arr, "iloc") else y_arr[tr]
            y_te = y_arr.iloc[te] if hasattr(y_arr, "iloc") else y_arr[te]

            # clone model if possible
            try:
                est = _sk_clone(model)
            except Exception:
                est = model

            fit_kwargs = {}
            if sample_weight is not None:
                sw_tr = sample_weight[tr]
                fit_kwargs["sample_weight"] = sw_tr
            try:
                est.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                est.fit(X_tr, y_tr)

            # predictions/scores
            y_pred = None
            y_score = None
            if hasattr(est, "predict"):
                try:
                    y_pred = est.predict(X_te)
                except Exception:
                    y_pred = None
            if hasattr(est, "predict_proba"):
                try:
                    proba = est.predict_proba(X_te)
                    y_score = (
                        proba[:, 1]
                        if _np.asarray(proba).ndim == 2 and _np.asarray(proba).shape[1] > 1
                        else _np.asarray(proba).ravel()
                    )
                except Exception:
                    y_score = None
            if y_score is None and hasattr(est, "decision_function"):
                try:
                    y_score = est.decision_function(X_te)
                except Exception:
                    y_score = None

            # compute metrics
            fold_metric_values: Dict[str, float] = {}
            for m in metric_list:
                try:
                    if m in ("accuracy",):
                        from sklearn.metrics import accuracy_score as _acc

                        if y_pred is not None:
                            val = float(_acc(y_te, y_pred))
                        else:
                            continue
                    elif m in ("f1", "f1_score"):
                        from sklearn.metrics import f1_score as _f1

                        if y_pred is not None:
                            val = float(_f1(y_te, y_pred))
                        else:
                            continue
                    elif m in ("precision",):
                        from sklearn.metrics import precision_score as _prec

                        if y_pred is not None:
                            val = float(_prec(y_te, y_pred))
                        else:
                            continue
                    elif m in ("recall",):
                        from sklearn.metrics import recall_score as _rec

                        if y_pred is not None:
                            val = float(_rec(y_te, y_pred))
                        else:
                            continue
                    elif m in ("sensitivity", "tpr", "recall_pos"):
                        from sklearn.metrics import recall_score as _rec

                        if y_pred is not None and binary_labels is not None:
                            pos_label = binary_labels[1]
                            val = float(_rec(y_te, y_pred, pos_label=pos_label, zero_division=0.0))
                        else:
                            continue
                    elif m in ("specificity", "tnr"):
                        from sklearn.metrics import recall_score as _rec

                        if y_pred is not None and binary_labels is not None:
                            neg_label = binary_labels[0]
                            val = float(_rec(y_te, y_pred, pos_label=neg_label, zero_division=0.0))
                        else:
                            continue
                    elif m in ("roc_auc", "auc"):
                        from sklearn.metrics import roc_auc_score as _auc

                        if y_score is not None:
                            val = float(_auc(y_te, y_score))
                        else:
                            continue
                    else:
                        # fallback to estimator.score as 'score'
                        if hasattr(est, "score") and m in ("score",):
                            val = float(est.score(X_te, y_te))
                        else:
                            continue
                    per_metric_scores[m].append(val)
                    fold_metric_values[m] = val
                except Exception:
                    # skip non-computable metric for this fold
                    pass

            fold_details.append(
                {
                    "fold": k,
                    "n_train": len(tr),
                    "n_val": len(te),
                    "metrics": fold_metric_values,
                }
            )

        # aggregate
        mean_scores: Dict[str, float] = {}
        std_scores: Dict[str, float] = {}
        conf_ints: Dict[str, Tuple[float, float]] = {}
        need_bootstrap_rng = (self.ci_method or "bootstrap").lower() in (
            "bootstrap",
            "boot",
            "bstrap",
        )
        rng = _np.random.default_rng(self.random_state) if need_bootstrap_rng else None

        for m, vals in per_metric_scores.items():
            arr = _np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            mean_scores[m] = float(arr.mean())
            std_scores[m] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            conf_ints[m] = self._compute_confidence_interval(arr, rng=rng)

        # leakage check (optional)
        leakage_check_map: Dict[str, bool] = self._basic_integrity_checks(
            X_arr, y_arr, groups=split_groups, splitter=splitter
        )
        recommendations: List[str] = []
        if leakage_checker is not None:
            try:
                leak_report = leakage_checker.check(X=X_arr, y=y_arr, groups=split_groups)
                leakage_check_map["has_leakage"] = not getattr(leak_report, "has_leakage", True)
                recs = getattr(leak_report, "recommendations", [])
                if recs:
                    recommendations.extend(recs)
            except Exception:
                leakage_check_map = {"has_leakage": True}

        # build detailed scores dict (arrays per metric)
        scores_dict = {
            m: np.asarray(v, dtype=float) for m, v in per_metric_scores.items() if len(v) > 0
        }

        ci_label = self._ci_method_label()

        result = ValidationResult(
            scores=scores_dict,
            mean_scores=mean_scores,
            std_scores=std_scores,
            confidence_intervals=conf_ints,
            ci_method=ci_label,
            ci_level=self.ci_level,
            fold_details=fold_details,
            leakage_check=leakage_check_map,
            recommendations=recommendations,
        )
        # Store last result for downstream consumers (e.g., reporting)
        try:
            self.last_result = result
        except Exception:
            pass
        return result

    def _setup_splitter(self):
        """Configure the appropriate CV splitter"""
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

        # Optional trustcv splitters for advanced strategies
        try:
            from .splitters import StratifiedGroupKFold as _TCVStratifiedGroupKFold
        except Exception:
            _TCVStratifiedGroupKFold = None
        _TCVHoldOut = _TCVRepeatedKFold = _TCVLOOCV = _TCVLPOCV = _TCVMonteCarloCV = (
            _TCVBootstrapValidation
        ) = None
        try:
            from .splitters import LOOCV as _TCVLOOCV
            from .splitters import LPOCV as _TCVLPOCV
            from .splitters import BootstrapValidation as _TCVBootstrapValidation
            from .splitters import HoldOut as _TCVHoldOut
            from .splitters import MonteCarloCV as _TCVMonteCarloCV
            from .splitters import RepeatedKFold as _TCVRepeatedKFold
        except Exception:
            pass

        # Normalize again in case an older object was created without normalization
        method_key = (
            self._normalize_method(self.method) if isinstance(self.method, str) else self.method
        )
        self.method = method_key

        if method_key == "kfold":
            self._cv_splitter = KFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
        elif method_key == "stratified_kfold":
            self._cv_splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
        elif method_key == "patient_grouped_kfold":
            self._cv_splitter = GroupKFold(n_splits=self.n_splits)
        elif method_key == "stratified_grouped_kfold":
            if _TCVStratifiedGroupKFold is None:
                raise ValueError(
                    "StratifiedGroupKFold is unavailable. Ensure trustcv.splitters is accessible."
                )
            self._cv_splitter = _TCVStratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        elif method_key == "temporal":
            self._cv_splitter = TimeSeriesSplit(n_splits=self.n_splits)
        elif method_key == "holdout":
            if _TCVHoldOut is None:
                raise ValueError(
                    "HoldOut splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            stratify_flag = bool(self.holdout_stratify)
            self._cv_splitter = _TCVHoldOut(
                test_size=self.holdout_test_size,
                random_state=self.random_state,
                stratify=True if stratify_flag else None,
            )
        elif method_key == "repeated_kfold":
            if _TCVRepeatedKFold is None:
                raise ValueError(
                    "RepeatedKFold splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            self._cv_splitter = _TCVRepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.repeated_kfold_repeats,
                random_state=self.random_state,
                stratify=self.repeated_kfold_stratify,
            )
        elif method_key == "loocv":
            if _TCVLOOCV is None:
                raise ValueError(
                    "LOOCV splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            self._cv_splitter = _TCVLOOCV()
        elif method_key == "lpocv":
            if _TCVLPOCV is None:
                raise ValueError(
                    "LPOCV splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            self._cv_splitter = _TCVLPOCV(p=self.lpocv_p)
        elif method_key == "monte_carlo":
            if _TCVMonteCarloCV is None:
                raise ValueError(
                    "MonteCarloCV splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            self._cv_splitter = _TCVMonteCarloCV(
                n_iterations=self.monte_carlo_iterations,
                test_size=self.monte_carlo_test_size,
                random_state=self.random_state,
            )
        elif method_key == "bootstrap":
            if _TCVBootstrapValidation is None:
                raise ValueError(
                    "BootstrapValidation splitter is unavailable. Ensure trustcv.splitters is installed."
                )
            self._cv_splitter = _TCVBootstrapValidation(
                n_iterations=self.bootstrap_validation_iterations,
                estimator=self.bootstrap_validation_estimator,
                random_state=self.random_state,
            )

        else:
            # Last-resort: accept common sklearn-style names literally
            name = str(self.method)
            namelow = name.lower()

            if namelow in ("stratifiedkfold", "stratified_kfold"):
                self._cv_splitter = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
                )
                self.method = "stratified_kfold"
                return

            if namelow in ("kfold",):
                self._cv_splitter = KFold(
                    n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
                )
                self.method = "kfold"
                return

            if namelow in ("groupkfold", "patientgroupedkfold", "groupedkfold", "group_kfold"):
                self._cv_splitter = GroupKFold(n_splits=self.n_splits)
                self.method = "patient_grouped_kfold"
                return

            if namelow in (
                "stratifiedgroupkfold",
                "stratified_groupkfold",
                "stratified_group_kfold",
                "stratifiedgroupedkfold",
            ):
                if _TCVStratifiedGroupKFold is None:
                    raise ValueError(
                        "StratifiedGroupKFold is unavailable. Ensure trustcv.splitters is accessible."
                    )
                self._cv_splitter = _TCVStratifiedGroupKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                )
                self.method = "stratified_grouped_kfold"
                return

            if namelow in ("timeseriessplit", "temporal", "time_series_split"):
                self._cv_splitter = TimeSeriesSplit(n_splits=self.n_splits)
                self.method = "temporal"
                return

            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use one of: 'kfold', 'stratified_kfold', 'patient_grouped_kfold', 'stratified_grouped_kfold', "
                f"'temporal', 'holdout', 'repeated_kfold', 'loocv', 'lpocv', 'monte_carlo', 'bootstrap' "
                f"(also accepts 'KFold', 'StratifiedKFold', 'GroupKFold', 'StratifiedGroupKFold', 'TimeSeriesSplit')."
            )

    @staticmethod
    def _validate_split_size(value: Union[float, int], name: str) -> Union[float, int]:
        """Validate hold-out style split size parameters."""
        if isinstance(value, float):
            if not 0 < value < 1:
                raise ValueError(f"{name} as float must be between 0 and 1.")
            return value
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{name} as int must be positive.")
            return value
        raise TypeError(f"{name} must be a float fraction or positive int.")

    def _normalize_method(self, method: str) -> str:
        """Normalize user-provided method names to canonical keys.

        Accepts variants like 'StratifiedKFold', 'stratified-kfold',
        'stratified_kfold', 'group_kfold', 'TimeSeriesSplit', etc.
        """
        if not isinstance(method, str):
            return method  # leave untouched if non-string

        import re

        # remove all non-alphanumeric and underscores, then lower
        # 'StratifiedKFold' -> 'stratifiedkfold'
        key = re.sub(r"[\W_]+", "", method).lower()

        mapping = {
            # IID / grouped
            "kfold": "kfold",
            "kfoldmedical": "kfold",
            "stratifiedkfold": "stratified_kfold",
            "stratifiedkfoldmedical": "stratified_kfold",
            "groupkfold": "patient_grouped_kfold",
            "groupedkfold": "patient_grouped_kfold",
            "patientgroupedkfold": "patient_grouped_kfold",
            "stratifiedgroupkfold": "stratified_grouped_kfold",
            "stratifiedgroupedkfold": "stratified_grouped_kfold",
            "stratifiedgroup_kfold": "stratified_grouped_kfold",
            "stratified_groupkfold": "stratified_grouped_kfold",
            "stratified_group_kfold": "stratified_grouped_kfold",
            "holdout": "holdout",
            "holdoutvalidation": "holdout",
            "holdoutsplit": "holdout",
            "traintestsplit": "holdout",
            "train_test_split": "holdout",
            "repeatedkfold": "repeated_kfold",
            "repeated_kfold": "repeated_kfold",
            "repeated-kfold": "repeated_kfold",
            "loocv": "loocv",
            "leaveoneout": "loocv",
            "leave_one_out": "loocv",
            "lpocv": "lpocv",
            "leavepout": "lpocv",
            "leave_p_out": "lpocv",
            "montecarlocv": "monte_carlo",
            "montecarlo": "monte_carlo",
            "monte_carlo": "monte_carlo",
            "bootstrapvalidation": "bootstrap",
            "bootstrap_cv": "bootstrap",
            "bootstrap": "bootstrap",
            # temporal
            "temporal": "temporal",
            "timeseriessplit": "temporal",
            "timeseries": "temporal",
        }
        return mapping.get(key, method)

    def _normalize_metric_list(self, metrics: Optional[Iterable[str]]) -> List[str]:
        """Normalize requested metric names and drop duplicates."""
        default = [
            "accuracy",
            "roc_auc",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
        ]
        if not metrics:
            return default.copy()
        alias_map = {
            "auc": "roc_auc",
            "rocauc": "roc_auc",
            "roc_auc": "roc_auc",
            "f1score": "f1",
            "f1_score": "f1",
            "score": "accuracy",
            "tpr": "sensitivity",
            "recall_pos": "sensitivity",
            "recall+": "sensitivity",
            "sensitivity": "sensitivity",
            "specificity": "specificity",
            "tnr": "specificity",
            "recall_neg": "specificity",
            "recall-": "specificity",
        }
        normalized: List[str] = []
        seen: set = set()
        for metric in metrics:
            if not isinstance(metric, str):
                continue
            key = metric.strip().lower()
            if not key:
                continue
            canonical = alias_map.get(key, key)
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized or default.copy()

    def fit_validate(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
        scoring: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Perform medical cross-validation

        Parameters:
        -----------
        model : sklearn estimator
            Model to validate
        X : array-like
            Features
        y : array-like
            Labels
        patient_ids : array-like, optional
            Patient identifiers for grouped splitting
        timestamps : array-like, optional
            Timestamps for temporal validation
        scoring : dict, optional
            Scoring metrics

        Returns:
        --------
        ValidationResult object with comprehensive metrics
        """
        # Default medical scoring metrics
        if scoring is None:
            scoring = self._get_medical_scoring()

        # Prepare groups for patient-level splitting
        groups = patient_ids if patient_ids is not None else None

        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=self._cv_splitter,
            groups=groups,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1,
        )

        # Calculate statistics (canonical keys without 'test_' prefix)
        mean_scores = {
            metric.replace("test_", ""): float(np.mean(scores))
            for metric, scores in cv_results.items()
            if metric.startswith("test_")
        }
        std_scores = {
            metric.replace("test_", ""): float(np.std(scores))
            for metric, scores in cv_results.items()
            if metric.startswith("test_")
        }
        # Backward-compatibility: also expose legacy 'test_*' keys alongside canonical ones
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                base = metric.replace("test_", "")
                if base in mean_scores and metric not in mean_scores:
                    mean_scores[metric] = mean_scores[base]
                if base in std_scores and metric not in std_scores:
                    std_scores[metric] = std_scores[base]

        # Calculate 95% confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(cv_results)
        # Alias common default scorer names for convenience
        if "score" in confidence_intervals and "accuracy" not in confidence_intervals:
            confidence_intervals["accuracy"] = confidence_intervals["score"]

        # Check for data leakage
        leakage_check = self._basic_integrity_checks(
            X, y, groups=patient_ids, splitter=self._cv_splitter
        )
        if self.check_leakage and patient_ids is not None:
            leakage_check.update(self._check_data_leakage(X, y, patient_ids, cv_results))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            mean_scores, std_scores, leakage_check, X.shape
        )

        # Prepare fold details
        fold_details = self._extract_fold_details(cv_results)

        # Normalize score keys so users can access without 'test_' prefix
        cv_results_normalized = dict(cv_results)
        for k, v in list(cv_results.items()):
            if k.startswith("test_"):
                cv_results_normalized[k.replace("test_", "")] = v
        if "score" in cv_results_normalized and "accuracy" not in cv_results_normalized:
            cv_results_normalized["accuracy"] = cv_results_normalized["score"]

        # Create result object
        ci_label = self._ci_method_label() if confidence_intervals else ""

        result = ValidationResult(
            scores=cv_results_normalized,
            mean_scores=mean_scores,
            std_scores=std_scores,
            confidence_intervals=confidence_intervals,
            ci_method=ci_label,
            ci_level=self.ci_level,
            fold_details=fold_details,
            leakage_check=leakage_check,
            recommendations=recommendations,
        )

        # Generate compliance report if needed
        if self.compliance:
            self._generate_compliance_report(result, model, X, y)

        return result

    def _get_medical_scoring(self) -> Dict[str, Any]:
        """Get medical-relevant scoring metrics"""
        from sklearn.metrics import make_scorer

        def sensitivity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0

        available = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "sensitivity": make_scorer(sensitivity_score),
            "specificity": make_scorer(specificity_score),
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }
        requested = self.metrics or list(available.keys())
        scoring: Dict[str, Any] = {}
        for metric in requested:
            scorer = available.get(metric)
            if scorer is None:
                warnings.warn(
                    f"Metric '{metric}' is not supported by fit_validate(); skipping.",
                    RuntimeWarning,
                )
                continue
            scoring[metric] = scorer
        return scoring or available

    def _ci_method_label(self) -> str:
        if not self.return_confidence_intervals:
            return ""
        method = (self.ci_method or "bootstrap").lower()
        if method in ("bootstrap", "boot", "bstrap"):
            return "bootstrap"
        if method in ("t", "t-interval", "t_interval", "student"):
            return "t-interval"
        return method

    def _compute_confidence_interval(
        self,
        values: np.ndarray,
        *,
        rng: Optional[np.random.Generator] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute confidence intervals over per-fold values."""
        arr = np.asarray(values, dtype=float)
        if arr.size <= 1 or not self.return_confidence_intervals:
            return (float("nan"), float("nan"))
        if alpha is None:
            alpha = 1.0 - float(self.ci_level or 0.95)
        method = (self.ci_method or "bootstrap").lower()
        if method in ("t", "t-interval", "t_interval", "student"):
            from scipy import stats

            mean = float(arr.mean())
            sem = stats.sem(arr)
            if np.isnan(sem):
                return (float("nan"), float("nan"))
            lo, hi = stats.t.interval(1 - alpha, len(arr) - 1, loc=mean, scale=sem)
            return (float(lo), float(hi))
        # default to bootstrap resampling
        n_bootstrap = max(int(self.n_bootstrap), 1)
        rng = rng or np.random.default_rng(self.random_state)
        boot = np.empty(n_bootstrap, dtype=float)
        n = arr.size
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(arr[idx].mean())
        lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100]).tolist()
        return (float(lo), float(hi))

    def _calculate_confidence_intervals(
        self, cv_results: Dict, alpha: Optional[float] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from cross_validate outputs."""
        if not self.return_confidence_intervals:
            return {}
        if alpha is None:
            alpha = 1.0 - float(self.ci_level or 0.95)
        confidence_intervals = {}
        use_bootstrap = (self.ci_method or "bootstrap").lower() in ("bootstrap", "boot", "bstrap")
        rng = np.random.default_rng(self.random_state) if use_bootstrap else None
        for metric in cv_results:
            if metric.startswith("test_"):
                scores = np.asarray(cv_results[metric], dtype=float)
                metric_name = metric.replace("test_", "")
                confidence_intervals[metric_name] = self._compute_confidence_interval(
                    scores, rng=rng, alpha=alpha
                )

        return confidence_intervals

    def _basic_integrity_checks(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        *,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
        splitter: Optional[BaseCrossValidator] = None,
    ) -> Dict[str, bool]:
        """Run lightweight integrity checks (balance, duplicates, group leakage)."""
        checks: Dict[str, bool] = {}
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        splitter = splitter or self._cv_splitter

        if self.check_leakage:
            # Duplicate sample check (DataFrame only to avoid expensive comparisons)
            checks["no_duplicate_samples"] = True
            if isinstance(X, pd.DataFrame):
                has_duplicates = bool(X.duplicated().any())
                checks["no_duplicate_samples"] = not has_duplicates
                if has_duplicates:
                    warnings.warn("Duplicate samples detected in dataset")
            groups_arr = None
            if groups is not None:
                groups_arr = (
                    groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
                )
                if len(groups_arr) != len(y_arr):
                    raise ValueError("groups/patient_ids length mismatch during integrity check.")
            if groups_arr is not None and splitter is not None:
                no_overlap_all = True
                try:
                    for train_idx, test_idx in splitter.split(X, y, groups_arr):
                        train_groups = set(np.unique(groups_arr[train_idx]))
                        test_groups = set(np.unique(groups_arr[test_idx]))
                        if train_groups.intersection(test_groups):
                            no_overlap_all = False
                            break
                except Exception:
                    no_overlap_all = False
                if not no_overlap_all:
                    warnings.warn(
                        "Group leakage detected: some group/patient IDs appear in both train and test folds."
                    )
                checks["no_patient_leakage"] = no_overlap_all

        if self.check_balance:
            balanced = True
            unique, counts = np.unique(y_arr, return_counts=True)
            if len(unique) == 2:
                ratio = counts.min() / counts.max()
                if ratio < 0.1:
                    balanced = False
                    warnings.warn(f"Severe class imbalance detected: {ratio:.2%} minority class")
            checks["balanced_classes"] = balanced

        return checks

    def _check_data_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Union[np.ndarray, pd.Series],
        cv_results: Dict,
    ) -> Dict[str, bool]:
        """Check for various types of data leakage"""
        return self._basic_integrity_checks(X, y, groups=patient_ids, splitter=self._cv_splitter)

    def _generate_recommendations(
        self,
        mean_scores: Dict[str, float],
        std_scores: Dict[str, float],
        leakage_check: Dict[str, bool],
        data_shape: Tuple,
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Check for high variance
        for metric, std in std_scores.items():
            if std > 0.1:  # High variance threshold
                recommendations.append(
                    f"High variance in {metric} ({std:.3f}). "
                    "Consider increasing sample size or using stratification."
                )

        # Sample size recommendations
        n_samples = data_shape[0]
        n_features = data_shape[1]
        if n_samples < 10 * n_features:
            recommendations.append(
                f"Low sample-to-feature ratio ({n_samples}/{n_features}). "
                "Consider feature selection or regularization."
            )

        # Leakage warnings
        if not all(leakage_check.values()):
            failed_checks = [k for k, v in leakage_check.items() if not v]
            recommendations.append(
                f"Data integrity issues detected: {', '.join(failed_checks)}. "
                "Review data preprocessing pipeline."
            )

        # Method-specific recommendations
        if self.method == "kfold" and not leakage_check.get("balanced_classes", True):
            recommendations.append("Consider using StratifiedKFold for imbalanced classes.")

        return recommendations

    def _extract_fold_details(self, cv_results: Dict) -> List[Dict]:
        """Extract detailed information for each fold"""
        fold_details = []
        n_folds = len(cv_results["test_accuracy"])

        for i in range(n_folds):
            fold_info = {
                "fold": i + 1,
                "train_score": cv_results["train_accuracy"][i],
                "test_score": cv_results["test_accuracy"][i],
                "fit_time": cv_results["fit_time"][i],
                "score_time": cv_results["score_time"][i],
            }
            fold_details.append(fold_info)

        return fold_details

    def _generate_compliance_report(
        self,
        result: ValidationResult,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """Generate regulatory compliance report"""
        if self.compliance == "FDA":
            # FDA-specific requirements
            report = {
                "device_description": str(model.__class__.__name__),
                "validation_method": self.method,
                "sample_size": len(y),
                "performance_metrics": result.to_dict(),
                "data_integrity": result.leakage_check,
                "random_seed": self.random_state,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Save FDA report
            with open("fda_validation_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)

            print("FDA validation report generated: fda_validation_report.json")

        elif self.compliance == "CE":
            # CE Mark requirements (simplified)
            print("CE compliance report generation (placeholder)")

    def suggest_best_method(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> str:
        """Suggest best CV method based on data characteristics"""
        n_samples = len(y)

        # Check for temporal data
        if timestamps is not None:
            return "temporal"

        # Check for grouped data
        if patient_ids is not None:
            unique_patients = len(np.unique(patient_ids))
            if unique_patients < n_samples:
                return "patient_grouped_kfold"

        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) > 1:
            ratio = counts.min() / counts.max()
            if ratio < 0.3:  # Imbalanced
                return "stratified_kfold"

        # Default
        return "kfold" if n_samples > 1000 else "stratified_kfold"


# --- Optional high-level nested CV runners ---
try:
    # Prefer public path; fallback to module path
    from .splitters import PurgedGroupTimeSeriesSplit as _PGTS
except Exception:
    try:
        from .splitters.temporal import PurgedGroupTimeSeriesSplit as _PGTS
    except Exception:
        _PGTS = None  # Will error if used without available splitter


class NestedGroupedCV:
    """
    Nested cross-validation for grouped data (patients/assets/etc.) with optional time-aware,
    leak-safe splitting via purge/embargo.

    Outer loop: unbiased evaluation on held-out groups.
    Inner loop: hyperparameter tuning on outer-train groups only.

    Parameters
    ----------
    n_splits_outer : int, default=5
        Number of outer folds.
    n_splits_inner : int, default=3
        Number of inner folds.
    group_exclusive : bool, default=True
        If True, any group that appears in the test fold is removed from the train for that fold.
    purge_gap : int, default=0
        Steps to purge before each test window from train (index-based, time-sorted).
    embargo_size : float, default=0.0
        Fraction of dataset length to embargo after each test window from train.
    scoring : {"average_precision","roc_auc"} or callable, default="average_precision"
        Metric used for inner hyperparameter selection.
    synthesize_timestamps : bool, default=True
        If True and timestamps is None, uses np.arange(n) as a monotonic timeline.
    """

    def __init__(
        self,
        n_splits_outer: int = 5,
        n_splits_inner: int = 3,
        *,
        group_exclusive: bool = True,
        purge_gap: int = 0,
        embargo_size: float = 0.0,
        scoring: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "average_precision",
        synthesize_timestamps: bool = True,
        mode: str = "time",
    ):
        """
        mode: "time" (default) uses PurgedGroupTimeSeriesSplit (contiguous windows).
              "group" uses sklearn's GroupKFold (balanced groups, no time windows).
        """

        self.n_splits_outer = int(n_splits_outer)
        self.n_splits_inner = int(n_splits_inner)
        self.group_exclusive = bool(group_exclusive)
        self.purge_gap = int(purge_gap)
        self.embargo_size = float(embargo_size)
        self.synthesize_timestamps = bool(synthesize_timestamps)
        self.mode = mode

        # scoring
        if isinstance(scoring, str):
            if scoring == "average_precision":
                self._score_fn = lambda yt, ys: __import__(
                    "sklearn.metrics"
                ).metrics.average_precision_score(yt, ys)
            elif scoring == "roc_auc":
                self._score_fn = lambda yt, ys: __import__("sklearn.metrics").metrics.roc_auc_score(
                    yt, ys
                )
            else:
                raise ValueError(
                    "scoring must be 'average_precision', 'roc_auc', or a callable(y_true, y_score)->float"
                )
        else:
            self._score_fn = scoring

        # Construct splitters once (same semantics inner/outer)
        if self.mode == "time":
            if _PGTS is None:
                raise ImportError("PurgedGroupTimeSeriesSplit not available in trustcv.splitters")
            self.outer_cv = _PGTS(
                n_splits=self.n_splits_outer,
                purge_gap=self.purge_gap,
                embargo_size=self.embargo_size,
                group_exclusive=self.group_exclusive,
            )
            self.inner_cv = _PGTS(
                n_splits=self.n_splits_inner,
                purge_gap=self.purge_gap,
                embargo_size=self.embargo_size,
                group_exclusive=self.group_exclusive,
            )
        elif self.mode == "group":
            # Exact GroupKFold semantics (no time windows)
            from sklearn.model_selection import GroupKFold

            self.outer_cv = GroupKFold(self.n_splits_outer)
            self.inner_cv = GroupKFold(self.n_splits_inner)
        else:
            raise ValueError("mode must be 'time' or 'group'")

    # utilities
    @staticmethod
    def _supports_arg(cv, name: str) -> bool:
        import inspect

        return name in inspect.signature(cv.split).parameters

    def _kw(self, cv, timestamps=None, groups=None) -> dict:
        kw = {}
        if self.mode == "time":
            if (timestamps is not None) and self._supports_arg(cv, "timestamps"):
                kw["timestamps"] = timestamps
            if (groups is not None) and self._supports_arg(cv, "groups"):
                kw["groups"] = groups
        else:  # group-only
            if groups is not None:
                kw["groups"] = groups
        return kw

    def fit_predict(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        param_grid: Dict[str, Iterable],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.base import clone
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            roc_auc_score,
        )
        from sklearn.model_selection import ParameterGrid

        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        n = len(X)
        assert X.shape[0] == y.shape[0] == groups.shape[0], "X, y, groups length mismatch"

        # timestamps handling
        if timestamps is None and self.synthesize_timestamps:
            timestamps = np.arange(n)
        elif timestamps is not None:
            timestamps = np.asarray(timestamps)
            assert timestamps.shape[0] == n, "timestamps length mismatch"

        rows = []

        # Outer loop
        for k, (tr, te) in enumerate(
            self.outer_cv.split(X, y, **self._kw(self.outer_cv, timestamps, groups)), 1
        ):
            X_tr, y_tr = X[tr], y[tr]
            ts_tr = timestamps[tr] if timestamps is not None else None
            grp_tr = groups[tr]

            # Inner loop hyperparameter tuning
            best_params, best_score = None, -np.inf
            for params in ParameterGrid(param_grid):
                scores = []
                for tr2, te2 in self.inner_cv.split(
                    X_tr, y_tr, **self._kw(self.inner_cv, ts_tr, grp_tr)
                ):
                    est = clone(estimator).set_params(**params)
                    est.fit(X_tr[tr2], y_tr[tr2])
                    proba = est.predict_proba(X_tr[te2])[:, 1]
                    scores.append(float(self._score_fn(y_tr[te2], proba)))
                mean_s = float(np.mean(scores)) if scores else -np.inf
                if mean_s > best_score:
                    best_score, best_params = mean_s, params
            if best_params is None:
                best_params = {}

            # Fit best on full outer-train; evaluate on outer-test
            best_est = clone(estimator).set_params(**best_params)
            best_est.fit(X_tr, y_tr)
            proba = best_est.predict_proba(X[te])[:, 1]
            pred = (proba >= 0.5).astype(int)

            rows.append(
                {
                    "fold": k,
                    "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                    "roc_auc": roc_auc_score(y[te], proba),
                    "auprc": average_precision_score(y[te], proba),
                    "f1": f1_score(y[te], pred),
                    "best_params": best_params,
                }
            )

        per_fold = pd.DataFrame(rows)
        summary = per_fold.drop(columns=["fold", "best_params"]).agg(["mean", "std"])
        return per_fold, summary


class NestedTemporalCV:
    """
    High-level nested CV for temporal (and optionally grouped) data.

    outer_cv: trustcv splitter for unbiased evaluation (e.g., TimeSeriesSplit, PurgedGroupTimeSeriesSplit)
    inner_cv: trustcv splitter for hyperparameter tuning (same family as outer)
    """

    def __init__(self, outer_cv, inner_cv):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

    def _supports_arg(self, cv, name: str) -> bool:
        import inspect

        return name in inspect.signature(cv.split).parameters

    def _cv_kwargs(self, cv, timestamps, groups):
        kw = {}
        if timestamps is not None and self._supports_arg(cv, "timestamps"):
            kw["timestamps"] = timestamps
        if groups is not None and self._supports_arg(cv, "groups"):
            kw["groups"] = groups
        return kw

    def fit_predict(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, Any],
        timestamps: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        scoring: str = "average_precision",
        return_per_fold: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.base import clone
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            roc_auc_score,
        )
        from sklearn.model_selection import ParameterGrid

        X = np.asarray(X)
        y = np.asarray(y)

        # Resolve tuning scorer: supports probability-based and label-based metrics
        def _resolve_scorer(sc):
            if callable(sc):
                return sc
            name = str(sc).lower()

            def _to_label(y_score):
                ys = y_score
                if hasattr(ys, "ndim") and ys.ndim > 1:
                    ys = ys[:, 1]
                thr = 0.5 if np.isfinite(ys).all() and ys.min() >= 0.0 and ys.max() <= 1.0 else 0.0
                return (ys >= thr).astype(int)

            if name in ("roc_auc", "auc"):
                return lambda yt, ys: roc_auc_score(yt, ys)
            if name in ("average_precision", "auprc", "pr_auc"):
                return lambda yt, ys: average_precision_score(yt, ys)
            if name in ("accuracy", "acc"):
                from sklearn.metrics import accuracy_score as _acc

                return lambda yt, ys: _acc(yt, _to_label(ys))
            if name in ("balanced_accuracy", "bacc", "balanced-accuracy"):
                return lambda yt, ys: balanced_accuracy_score(yt, _to_label(ys))
            if name in ("f1", "f1_score"):
                return lambda yt, ys: f1_score(yt, _to_label(ys))
            if name in ("precision", "ppv"):
                from sklearn.metrics import precision_score as _prec

                return lambda yt, ys: _prec(yt, _to_label(ys))
            if name in ("recall", "tpr", "sensitivity"):
                from sklearn.metrics import recall_score as _rec

                return lambda yt, ys: _rec(yt, _to_label(ys))
            raise ValueError(
                "Unsupported scoring. Use 'roc_auc', 'average_precision', 'accuracy',"
                " 'balanced_accuracy', 'f1', 'precision', 'recall', or a callable."
            )

        score_fn = _resolve_scorer(scoring)

        rows = []
        outer_kw = self._cv_kwargs(self.outer_cv, timestamps, groups)

        for k, (tr, te) in enumerate(self.outer_cv.split(X, y, **outer_kw), 1):
            X_tr, y_tr = X[tr], y[tr]
            ts_tr = timestamps[tr] if timestamps is not None else None
            gp_tr = groups[tr] if groups is not None else None

            # inner tuning
            inner_kw = self._cv_kwargs(self.inner_cv, ts_tr, gp_tr)
            best_params, best_score = None, -np.inf
            for params in ParameterGrid(param_grid):
                scores = []
                for tr2, te2 in self.inner_cv.split(X_tr, y_tr, **inner_kw):
                    est = clone(estimator).set_params(**params)
                    est.fit(X_tr[tr2], y_tr[tr2])
                    if hasattr(est, "predict_proba"):
                        proba = est.predict_proba(X_tr[te2])[:, 1]
                    else:
                        proba = est.decision_function(X_tr[te2])
                    scores.append(float(score_fn(y_tr[te2], proba)))
                mean_s = float(np.mean(scores)) if scores else -np.inf
                if mean_s > best_score:
                    best_score, best_params = mean_s, params
            # Fallback to estimator defaults if inner loop produced no valid scores
            if best_params is None:
                best_params = {}

            # fit best on full outer-train
            best_est = clone(estimator).set_params(**best_params)
            best_est.fit(X_tr, y_tr)
            if hasattr(best_est, "predict_proba"):
                proba = best_est.predict_proba(X[te])[:, 1]
            else:
                proba = best_est.decision_function(X[te])
            pred = (
                (proba >= 0.5).astype(int) if proba.ndim == 1 else (proba[:, 1] >= 0.5).astype(int)
            )

            rows.append(
                {
                    "fold": k,
                    "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                    "roc_auc": roc_auc_score(y[te], proba),
                    "auprc": average_precision_score(y[te], proba),
                    "f1": f1_score(y[te], pred),
                    "best_params": best_params,
                }
            )

        per_fold = pd.DataFrame(rows)
        summary = per_fold.drop(columns=["fold", "best_params"]).agg(["mean", "std"])
        return per_fold if return_per_fold else summary, summary


import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array


class EnvironmentalHealthCV(BaseCrossValidator):
    """Spatial block cross-validation tailored for environmental-health datasets."""

    def __init__(
        self,
        spatial_blocks=5,
        shuffle=True,
        random_state=None,
        *,
        environmental_vars=None,
        buffer_config=None,
        coordinates=None,
        timestamps=None,
    ):
        self.spatial_blocks = spatial_blocks
        self.shuffle = shuffle
        self.random_state = random_state
        self.environmental_vars = environmental_vars
        self.buffer_config = buffer_config or {}
        self._default_coordinates = coordinates
        self.timestamps = timestamps

    def get_n_splits(self, *args, **kwargs):
        return self.spatial_blocks

    def split(self, X, y=None, groups=None, **kwargs):
        # Accept either a feature matrix or a sequence of indices from the runner.
        if isinstance(X, range):
            indices = np.arange(len(X), dtype=int)
            feature_matrix = None
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                indices = X_arr.astype(int)
                feature_matrix = None
            else:
                feature_matrix = check_array(X_arr, accept_sparse=False)
                indices = np.arange(feature_matrix.shape[0])

        n_samples = indices.shape[0]

        # Coordinates priority: per-call -> default -> first two features -> environmental covariates.
        coords = kwargs.get("coordinates", self._default_coordinates)

        if coords is None and feature_matrix is not None and feature_matrix.shape[1] >= 2:
            coords = feature_matrix[:, :2]

        env_vars = kwargs.get("environmental_vars", self.environmental_vars)
        buffer_cfg = kwargs.get("buffer_config", self.buffer_config)
        timestamps = kwargs.get("timestamps", self.timestamps)
        _ = (buffer_cfg, timestamps)  # reserved for future enhancements

        if coords is None and env_vars:
            # Use environmental covariates as proxy coordinates (deterministic order).
            ordered_keys = sorted(env_vars.keys())
            env_matrix = np.column_stack([np.asarray(env_vars[k]) for k in ordered_keys])
            env_matrix = check_array(env_matrix, ensure_2d=True)
            if env_matrix.shape[0] != n_samples:
                raise ValueError("Environmental variables must align with number of samples.")
            # If only one covariate, pad a zero column to form 2-D coordinates
            if env_matrix.shape[1] == 1:
                coords = np.column_stack([env_matrix[:, 0], np.zeros_like(env_matrix[:, 0])])
            else:
                coords = env_matrix[:, :2]

        if coords is None:
            raise ValueError(
                "EnvironmentalHealthCV requires `coordinates`, environmental covariates, "
                "or a feature matrix with at least two columns."
            )

        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be shaped (n_samples, 2).")
        if coords.shape[0] != n_samples:
            raise ValueError("Coordinate count does not match number of samples.")

        # Build spatial blocks using a simple grid.
        x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), self.spatial_blocks + 1)
        y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), self.spatial_blocks + 1)
        x_bins = np.digitize(coords[:, 0], x_edges[1:-1], right=False)
        y_bins = np.digitize(coords[:, 1], y_edges[1:-1], right=False)
        block_ids = x_bins * self.spatial_blocks + y_bins

        unique_blocks = np.unique(block_ids)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(unique_blocks)

        for block in unique_blocks:
            mask = block_ids == block
            if not mask.any():
                continue
            yield indices[~mask], indices[mask]


# ensure it's exported
try:
    __all__.append("EnvironmentalHealthCV")
except NameError:
    __all__ = ["EnvironmentalHealthCV"]

# Backward-compatibility alias for deprecated name
import warnings as _tcv_warnings


class MedicalValidator(TrustCVValidator):  # deprecated alias
    def __init__(self, *args, **kwargs):
        _tcv_warnings.warn(
            "MedicalValidator is deprecated and will be removed in v2.0. Use TrustCVValidator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Ensure module exports include both names
try:
    __all__
except NameError:  # pragma: no cover
    __all__ = []

for _name in ("TrustCVValidator", "MedicalValidator"):
    if _name not in __all__:
        __all__.append(_name)
