"""
High-level helpers for generating regulatory reports from UniversalCVRunner results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..metrics import ClinicalMetrics
from .regulatory_report import RegulatoryReport


class UniversalRegulatoryReport:
    """
    Convenience wrapper that builds a RegulatoryReport directly from CVResults.

    Example
    -------
    >>> report_path = UniversalRegulatoryReport.from_runner(
    ...     runner_results=res,
    ...     model=model,
    ...     data=(X, y),
    ...     output_path="reports/regulatory.html",
    ... )
    """

    DEFAULT_METRICS_PRIORITY: Sequence[str] = (
        "balanced_accuracy",
        "accuracy",
        "roc_auc",
        "f1",
        "score",
    )

    @classmethod
    def from_runner(
        cls,
        *,
        runner_results,
        model: Any,
        data: Tuple[Any, Any],
        output_path: str,
        clinical_metrics: Optional[Dict[str, Any]] = None,
        report_format: str = "html",
        model_name: Optional[str] = None,
        model_version: str = "1.0.0",
        manufacturer: str = "Unknown",
        intended_use: str = "Clinical decision support via machine learning.",
        compliance_standard: str = "FDA",
        project_name: Optional[str] = None,
        metric_priority: Optional[Sequence[str]] = None,
        positive_threshold: float = 0.5,
        n_patients: Optional[int] = None,
        demographics: Optional[Dict[str, Any]] = None,
        data_sources: Optional[Iterable[str]] = None,
    ) -> str:
        """
        Build and save a regulatory report from UniversalCVRunner results.
        """
        report, _ = cls._build_report(
            runner_results=runner_results,
            model=model,
            data=data,
            clinical_metrics=clinical_metrics,
            report_format=report_format,
            model_name=model_name,
            model_version=model_version,
            manufacturer=manufacturer,
            intended_use=intended_use,
            compliance_standard=compliance_standard,
            project_name=project_name,
            metric_priority=metric_priority,
            positive_threshold=positive_threshold,
            n_patients=n_patients,
            demographics=demographics,
            data_sources=data_sources,
        )
        output_path = str(Path(output_path))
        return report.generate_regulatory_report(
            output_path=output_path,
            format=report_format,
        )

    @classmethod
    def clinical_report_from_runner(
        cls,
        *,
        runner_results,
        model: Any,
        data: Tuple[Any, Any],
        clinical_metrics: Optional[Dict[str, Any]] = None,
        output_path: str,
        report_format: str = "html",
        model_name: Optional[str] = None,
        model_version: str = "1.0.0",
        manufacturer: str = "Unknown",
        intended_use: str = "Clinical decision support via machine learning.",
        compliance_standard: str = "FDA",
        project_name: Optional[str] = None,
        metric_priority: Optional[Sequence[str]] = None,
        positive_threshold: float = 0.5,
        n_patients: Optional[int] = None,
        demographics: Optional[Dict[str, Any]] = None,
        data_sources: Optional[Iterable[str]] = None,
    ) -> str:
        """
        Build the dedicated clinical performance report from UniversalCVRunner results.
        """
        report, perf_metrics = cls._build_report(
            runner_results=runner_results,
            model=model,
            data=data,
            clinical_metrics=clinical_metrics,
            report_format=report_format,
            model_name=model_name,
            model_version=model_version,
            manufacturer=manufacturer,
            intended_use=intended_use,
            compliance_standard=compliance_standard,
            project_name=project_name,
            metric_priority=metric_priority,
            positive_threshold=positive_threshold,
            n_patients=n_patients,
            demographics=demographics,
            data_sources=data_sources,
        )
        if not perf_metrics:
            raise ValueError(
                "Clinical metrics unavailable. Ensure predictions/probabilities are stored "
                "or pass `clinical_metrics` explicitly."
            )
        output_path = str(Path(output_path))
        return report.clinicalperformancereport(
            metrics=perf_metrics,
            output_path=output_path,
            format=report_format,
        )

    # ----- helpers -----
    @classmethod
    def _build_report(
        cls,
        *,
        runner_results,
        model: Any,
        data: Tuple[Any, Any],
        clinical_metrics: Optional[Dict[str, Any]],
        model_name: Optional[str],
        model_version: str,
        manufacturer: str,
        intended_use: str,
        compliance_standard: str,
        project_name: Optional[str],
        metric_priority: Optional[Sequence[str]],
        positive_threshold: float,
        n_patients: Optional[int],
        demographics: Optional[Dict[str, Any]],
        data_sources: Optional[Iterable[str]],
        report_format: Optional[str] = None,
    ) -> Tuple[RegulatoryReport, Optional[Dict[str, Any]]]:
        X, y = cls._unpack_xy(data)
        model_name = model_name or cls._infer_model_name(model)

        metadata = cls._coerce_metadata(getattr(runner_results, "metadata", None))
        report = RegulatoryReport(
            model_name=model_name,
            model_version=model_version,
            manufacturer=manufacturer,
            intended_use=intended_use,
            compliance_standard=compliance_standard,
            project_name=project_name,
        )

        n_samples, n_features = cls._infer_dataset_shape(X)
        class_distribution = cls._compute_class_distribution(y)

        dataset_info = {
            "n_patients": n_patients if n_patients is not None else n_samples,
            "n_samples": n_samples,
            "n_features": n_features,
            "demographics": demographics or {},
            "data_sources": list(data_sources) if data_sources is not None else [],
            "class_distribution": class_distribution,
        }
        report.add_dataset_info(**dataset_info)

        fold_scores, metric_name = cls._extract_fold_scores(
            runner_results, priority=metric_priority or cls.DEFAULT_METRICS_PRIORITY
        )
        inferred_method = (
            metadata.get("cv_method")
            or metadata.get("method")
            or getattr(runner_results, "method", None)
            or "Unknown"
        )
        inferred_splits = (
            metadata.get("n_splits")
            or getattr(runner_results, "n_splits", None)
            or len(fold_scores)
        )
        report.add_cv_results(
            method=inferred_method,
            n_splits=int(inferred_splits or len(fold_scores)),
            scores=fold_scores,
        )

        perf_metrics = clinical_metrics or cls._compute_clinical_metrics(
            runner_results,
            y=y,
            threshold=positive_threshold,
        )
        if perf_metrics:
            report.performance_metrics = perf_metrics
        return report, perf_metrics

    @staticmethod
    def _unpack_xy(data: Tuple[Any, ...]) -> Tuple[Any, Any]:
        if not isinstance(data, tuple) or len(data) < 2:
            raise ValueError("data must be a tuple like (X, y) or (X, y, groups)")
        return data[0], data[1]

    @staticmethod
    def _infer_model_name(model: Any) -> str:
        if hasattr(model, "steps"):
            # sklearn pipeline
            names = [step.__class__.__name__ for _, step in getattr(model, "steps", [])]
            return " -> ".join(names) or model.__class__.__name__
        return getattr(model, "__class__", type("Anonymous", (), {})).__name__

    @staticmethod
    def _infer_dataset_shape(X: Any) -> Tuple[int, int]:
        if hasattr(X, "shape") and len(X.shape) >= 2:
            return int(X.shape[0]), int(X.shape[1])
        if hasattr(X, "__len__"):
            n_samples = len(X)
            try:
                first = X[0]
                n_features = len(first)
            except Exception:
                n_features = 0
            return int(n_samples), int(n_features)
        raise ValueError("Unable to infer dataset shape from X")

    @staticmethod
    def _extract_fold_scores(runner_results, priority: Sequence[str]) -> Tuple[list, str]:
        raw_scores = getattr(runner_results, "scores", None) or []
        if isinstance(raw_scores, dict):
            usable_scores = UniversalRegulatoryReport._expand_score_dict(raw_scores)
        else:
            usable_scores = raw_scores

        scores = []
        metric_name = None
        for candidate in priority:
            collected = []
            for fold in usable_scores or []:
                if not isinstance(fold, dict):
                    continue
                value = fold.get(candidate)
                if value is None:
                    continue
                val = np.asarray(value).ravel()
                if val.size == 0:
                    continue
                collected.append(float(val.mean()))
            if collected:
                scores = collected
                metric_name = candidate
                break
        if not scores and usable_scores:
            # fall back to first numeric entry per fold
            for fold in usable_scores:
                if not isinstance(fold, dict):
                    continue
                for key, value in fold.items():
                    try:
                        val = float(np.asarray(value).ravel()[0])
                        scores.append(val)
                        metric_name = key
                        break
                    except Exception:
                        continue
                if scores:
                    break
        return scores, metric_name or "score"

    @staticmethod
    def _compute_clinical_metrics(runner_results, y, threshold: float) -> Optional[Dict]:
        indices = getattr(runner_results, "indices", None)
        if not indices:
            return None
        from . import __all__  # noqa: F401  (keeps import order)

        y_true_all = []
        y_pred_all = []
        y_proba_all = []

        for (train_idx, val_idx), preds, probas in zip(
            indices,
            runner_results.predictions or [None] * len(indices),
            runner_results.probabilities or [None] * len(indices),
        ):
            y_slice = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
            y_true_all.append(np.asarray(y_slice))

            if preds is not None:
                y_pred_all.append(np.asarray(preds))
            elif probas is not None:
                arr = np.asarray(probas)
                if arr.ndim == 2:
                    arr = arr[:, 1]
                y_pred_all.append((arr >= threshold).astype(int))

            if probas is not None:
                arr = np.asarray(probas)
                if arr.ndim == 2:
                    arr = arr[:, 1]
                y_proba_all.append(arr)

        if not y_pred_all:
            return None

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_proba = np.concatenate(y_proba_all) if y_proba_all else None

        metrics = ClinicalMetrics().calculate_all(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        return metrics

    @staticmethod
    def _expand_score_dict(scores_dict: Dict[str, Any]) -> list:
        """Convert ValidationResult-style score dicts into per-fold dictionaries."""
        fold_data: list = []
        max_len = 0
        arrays: Dict[str, np.ndarray] = {}
        for key, values in scores_dict.items():
            try:
                arr = np.asarray(values)
            except Exception:
                continue
            if arr.ndim == 0:
                continue
            arr = arr.ravel()
            if arr.size == 0:
                continue
            arrays[key] = arr
            max_len = max(max_len, arr.size)

        if max_len == 0:
            return fold_data

        fold_data = [dict() for _ in range(max_len)]
        for key, arr in arrays.items():
            for idx, val in enumerate(arr):
                fold_data[idx][key] = float(val)
        return fold_data

    @staticmethod
    def _coerce_metadata(metadata_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(metadata_obj, dict):
            return metadata_obj
        return {}

    @staticmethod
    def _compute_class_distribution(y: Any) -> Dict[str, Dict[str, float]]:
        try:
            if hasattr(y, "value_counts"):
                counts = y.value_counts()
            else:
                unique, cnts = np.unique(np.asarray(y), return_counts=True)
                counts = dict(zip(unique, cnts))
        except Exception:
            return {}

        if not isinstance(counts, dict):
            counts = counts.to_dict()

        total = float(sum(counts.values()))
        distribution = {}
        for label, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0.0
            distribution[str(label)] = {
                "count": int(count),
                "percentage": pct,
            }
        return distribution
