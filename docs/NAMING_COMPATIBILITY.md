# Naming Compatibility Map for `trustcv`

This document maps current exported names to **canonical** names for consistency.

| Current | Canonical | Rationale | Action |
|---|---|---|---|
| `MedicalValidator` | `TrustCVValidator` | High-level evaluator; deprecated alias remains | Use `TrustCVValidator` |
| `DataLeakageChecker` | `LeakageChecker` | Shorter naming, same meaning | Add alias; keep old |
| `BalanceChecker` | `ClassBalanceChecker` | Explicitly class balance | Add alias; keep old |
| `ClinicalMetrics` | `ClinicalMetrics` | Clear domain term | Keep |
| `UniversalCVRunner` | `UniversalCVRunner` | Framework-agnostic runner | Keep |
| `CVResults` | `CVResults` | Result container | Keep |
| `CVCallback` | `CVCallback` | Callback API | Keep |
| `EarlyStopping` | `EarlyStopping` | Standard term | Keep |
| `ModelCheckpoint` | `ModelCheckpoint` | Standard term | Keep |
| `ProgressLogger` | `ProgressLogger` | Progress logging | Keep |
| `HoldOut` | `HoldoutSplit` | Class-style naming | Add alias; keep old |
| `KFoldMedical` | `KFold` | Match sklearn | Add alias; deprecate old |
| `StratifiedKFoldMedical` | `StratifiedKFold` | Match sklearn | Add alias; deprecate old |
| `RepeatedKFold` | `RepeatedKFold` | Already standard | Keep |
| `LOOCV` | `LeaveOneOut` | Match sklearn | Add alias; deprecate old |
| `LPOCV` | `LeavePOut` | Match sklearn | Add alias; deprecate old |
| `BootstrapValidation` | `BootstrapCV` | Terse and clear | Add alias; keep old |
| `MonteCarloCV` | `MonteCarloCV` | Recognizable | Keep |
| `NestedCV` | `NestedCV` | Recognizable | Keep |
| `GroupKFoldMedical` | `GroupKFold` | Match sklearn | Add alias; deprecate old |
| `StratifiedGroupKFold` | `StratifiedGroupKFold` | Recognizable | Keep |
| `LeaveOneGroupOut` | `LeaveOneGroupOut` | Matches sklearn | Keep |
| `RepeatedGroupKFold` | `RepeatedGroupKFold` | Clear | Keep |
| `NestedGroupedCV` | `NestedGroupCV` | Shorter alias | Add alias; keep old |
| `HierarchicalGroupKFold` | `HierarchicalGroupKFold` | Distinct behavior | Keep |
| `TimeSeriesSplit` | `TimeSeriesSplit` | Matches sklearn | Keep |
| `BlockedTimeSeries` | `BlockedTimeSeriesSplit` | Clarify as splitter | Add alias; keep old |
| `RollingWindowCV` | `RollingWindowSplit` | Clarify as splitter | Add alias; keep old |
| `ExpandingWindowCV` | `ExpandingWindowSplit` | Clarify as splitter | Add alias; keep old |
| `PurgedKFoldCV` | `PurgedKFold` | Finance literature | Add alias; keep old |
| `CombinatorialPurgedCV` | `CombinatorialPurgedKFold` | Finance literature | Add alias; keep old |
| `PurgedGroupTimeSeriesSplit` | `PurgedGroupTimeSeriesSplit` | Specific & clear | Keep |
| `NestedTemporalCV` | `NestedTemporalCV` | Clear | Keep |
| `SpatialBlockCV` | `SpatialBlockSplit` | Clarify as splitter | Add alias; keep old |
| `BufferedSpatialCV` | `BufferedSpatialSplit` | Clarify as splitter | Add alias; keep old |
| `SpatiotemporalBlockCV` | `SpatiotemporalBlockSplit` | Clarify as splitter | Add alias; keep old |
| `EnvironmentalHealthCV` | `EnvironmentalHealthSplit` | Neutral splitter naming | Add alias; keep old |
