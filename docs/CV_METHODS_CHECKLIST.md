# CV Methods Checklist

Use this checklist to select and validate cross‑validation strategies in trustcv. It is designed for regulatory‑minded workflows where you must justify your split strategy and document leakage safeguards.

Last updated: 2026-02-05

## How To Use
1. Read the dataset context section and mark every item that applies.
2. Pick the appropriate CV family (IID, Grouped, Temporal, Spatial, Multilabel).
3. Mark each method you evaluated.
4. Record the final method chosen and the rationale.

## Dataset Context
- [ ] Independent samples (no shared subjects, sites, or time)
- [ ] Grouped by patient/subject/site
- [ ] Temporal ordering matters
- [ ] Spatial correlation or geographic clustering exists
- [ ] Multiple labels per sample (multilabel)
- [ ] Class imbalance is severe
- [ ] High risk of data leakage

## Leakage Safeguards
- [ ] Confirmed patient/group identifiers are present
- [ ] Leakage checker run and reviewed
- [ ] Split performed before any target‑dependent preprocessing
- [ ] No shared entities across train/test (patients, scans, sessions)
- [ ] Temporal splits respect chronology

## IID Methods
- [ ] HoldOut
- [ ] KFoldMedical
- [ ] StratifiedKFoldMedical
- [ ] RepeatedKFold
- [ ] LOOCV
- [ ] LPOCV
- [ ] BootstrapValidation
- [ ] MonteCarloCV
- [ ] NestedCV

## Grouped Methods
- [ ] GroupKFoldMedical
- [ ] StratifiedGroupKFold
- [ ] LeaveOneGroupOut
- [ ] LeavePGroupsOut
- [ ] RepeatedGroupKFold
- [ ] NestedGroupedCV
- [ ] HierarchicalGroupKFold
- [ ] MultilabelStratifiedGroupKFold

## Temporal Methods
- [ ] TimeSeriesSplit
- [ ] BlockedTimeSeries
- [ ] RollingWindowCV
- [ ] ExpandingWindowCV
- [ ] PurgedKFoldCV
- [ ] CombinatorialPurgedCV
- [ ] PurgedGroupTimeSeriesSplit
- [ ] NestedTemporalCV

## Spatial Methods
- [ ] SpatialBlockCV
- [ ] BufferedSpatialCV
- [ ] SpatiotemporalBlockCV
- [ ] EnvironmentalHealthCV

## Multilabel Methods
- [ ] MultilabelStratifiedKFold
- [ ] MultilabelStratifiedGroupKFold

## Final Decision
Selected method:

Rationale:

Known limitations:

## Audit Notes
Date:

Reviewer:

Version:

