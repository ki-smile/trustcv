"""
High-level helpers for generating regulatory reports from UniversalCVRunner results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .regulatory_report import RegulatoryReport
from ..metrics import ClinicalMetrics


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
        X, y = cls._unpack_xy(data)
        model_name = model_name or cls._infer_model_name(model)

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
        report.add_cv_results(
            method=runner_results.metadata.get("cv_method", "Unknown"),
            n_splits=runner_results.metadata.get(
                "n_splits", len(runner_results.scores or [])
            ),
            scores=fold_scores,
        )

        perf_metrics = clinical_metrics or cls._compute_clinical_metrics(
            runner_results,
            y=y,
            threshold=positive_threshold,
        )
        if perf_metrics:
            report.performance_metrics = perf_metrics

        output_path = str(Path(output_path))
        return report.generate_regulatory_report(
            output_path=output_path,
            format=report_format,
        )

    # ----- helpers -----
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
        scores = []
        metric_name = None
        for candidate in priority:
            collected = []
            for fold in runner_results.scores or []:
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
        if not scores and runner_results.scores:
            # fall back to first numeric entry per fold
            for fold in runner_results.scores:
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
        if not runner_results.indices:
            return None
        from . import __all__  # noqa: F401  (keeps import order)

        indices = runner_results.indices or []
        if not indices:
            return None

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
