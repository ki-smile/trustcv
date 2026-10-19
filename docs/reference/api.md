# API reference

Generated from the docstrings of the installed source. Import everything from the top-level package, e.g. `from trustcv import TrustCV, StratifiedGroupKFold`; deep-learning runners live in `trustcv.frameworks.*` and need their framework installed.

## Validation

::: trustcv.validators.TrustCVValidator

::: trustcv.validators.ValidationResult

::: trustcv.checks.CheckResult

::: trustcv.advisor.recommend_cv

::: trustcv.advisor.CVRecommendation

## Splitters: independent samples

::: trustcv.splitters.iid.HoldOut

::: trustcv.splitters.iid.KFoldMedical

::: trustcv.splitters.iid.StratifiedKFoldMedical

::: trustcv.splitters.iid.RepeatedKFold

::: trustcv.splitters.iid.LOOCV

::: trustcv.splitters.iid.LPOCV

::: trustcv.splitters.iid.BootstrapValidation

::: trustcv.splitters.iid.MonteCarloCV

::: trustcv.splitters.iid.NestedCV

## Splitters: grouped

::: trustcv.splitters.grouped.GroupKFoldMedical

::: trustcv.splitters.grouped.StratifiedGroupKFold

::: trustcv.splitters.grouped.LeaveOneGroupOut

::: trustcv.splitters.grouped.LeavePGroupsOut

::: trustcv.splitters.grouped.RepeatedGroupKFold

::: trustcv.splitters.grouped.HierarchicalGroupKFold

::: trustcv.splitters.grouped.NestedGroupedCV

## Splitters: temporal

::: trustcv.splitters.temporal.TimeSeriesSplit

::: trustcv.splitters.temporal.BlockedTimeSeries

::: trustcv.splitters.temporal.RollingWindowCV

::: trustcv.splitters.temporal.ExpandingWindowCV

::: trustcv.splitters.temporal.PurgedKFoldCV

::: trustcv.splitters.temporal.CombinatorialPurgedCV

::: trustcv.splitters.temporal.PurgedGroupTimeSeriesSplit

::: trustcv.validators.NestedTemporalCV

## Splitters: spatial

::: trustcv.splitters.spatial.SpatialBlockCV

::: trustcv.splitters.spatial.BufferedSpatialCV

::: trustcv.splitters.spatial.SpatiotemporalBlockCV

::: trustcv.splitters.spatial.EnvironmentalHealthCV

## Splitters: multilabel

::: trustcv.splitters.multilabel.MultilabelStratifiedKFold

::: trustcv.splitters.multilabel_group.MultilabelStratifiedGroupKFold

## Leakage and balance checks

::: trustcv.checkers.leakage.DataLeakageChecker

::: trustcv.checkers.leakage.LeakageReport

::: trustcv.checkers.balance.BalanceChecker

## Metrics

::: trustcv.metrics.clinical.ClinicalMetrics

::: trustcv.metrics.diagnostics.check_fold_metric_feasibility

## Deep-learning runners

::: trustcv.core.runner.UniversalCVRunner

::: trustcv.core.base.CVResults

::: trustcv.frameworks.tensorflow.KerasCVRunner

::: trustcv.frameworks.pytorch.TorchCVRunner

::: trustcv.frameworks.monai.MONAICVRunner

::: trustcv.frameworks.jax.JAXCVRunner

::: trustcv.frameworks.tensorflow_sklearn.KerasBaseWrap

::: trustcv.frameworks.tensorflow_sklearn.KerasClassifierWrap

::: trustcv.frameworks.tensorflow_sklearn.KerasRegressorWrap

::: trustcv.core.inner_split.inner_validation_split

## Callbacks

::: trustcv.core.callbacks.CVCallback

::: trustcv.core.callbacks.EarlyStopping

::: trustcv.core.callbacks.ModelCheckpoint

::: trustcv.core.callbacks.ProgressLogger

::: trustcv.core.callbacks.ClassDistributionLogger

::: trustcv.core.callbacks.LeakageDetectionCallback

::: trustcv.core.callbacks.RegulatoryComplianceLogger

## Datasets

::: trustcv.datasets.loaders.load_heart_disease

::: trustcv.datasets.loaders.load_diabetic_readmission

::: trustcv.datasets.loaders.load_cancer_imaging

::: trustcv.datasets.loaders.generate_synthetic_ehr

::: trustcv.datasets.loaders.generate_temporal_patient_data

## Aliases

Alternative names kept for compatibility. They refer to the same objects.

| Alias | Same as |
| --- | --- |
| `TrustCV` | `TrustCVValidator` |
| `MedicalValidator` | `TrustCVValidator (deprecated)` |
| `LeakageChecker` | `DataLeakageChecker` |
| `ClassBalanceChecker` | `BalanceChecker` |
| `KFold` | `KFoldMedical` |
| `StratifiedKFold` | `StratifiedKFoldMedical` |
| `LeaveOneOut` | `LOOCV` |
| `LeavePOut` | `LPOCV` |
| `GroupKFold` | `GroupKFoldMedical` |
| `PatientGroupKFold` | `GroupKFoldMedical` |
| `TemporalClinical` | `TimeSeriesSplit` |
| `BlockedTimeSeriesSplit` | `BlockedTimeSeries` |
| `RollingWindowSplit` | `RollingWindowCV` |
| `ExpandingWindowSplit` | `ExpandingWindowCV` |
| `PurgedKFold` | `PurgedKFoldCV` |
| `CombinatorialPurgedKFold` | `CombinatorialPurgedCV` |
| `SpatialBlockSplit` | `SpatialBlockCV` |
| `BufferedSpatialSplit` | `BufferedSpatialCV` |
| `SpatiotemporalBlockSplit` | `SpatiotemporalBlockCV` |
| `EnvironmentalHealthSplit` | `EnvironmentalHealthCV` |
| `KerasSkWrap` | `KerasClassifierWrap` |
