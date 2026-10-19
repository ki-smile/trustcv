---
description: See how trustcv splitters divide data into training and test folds for independent, grouped, temporal and spatial data, and explore simulated scenarios where the wrong split flatters a model.
---

<link rel="stylesheet" href="../assets/css/learn.css">

# Interactive visualisations

A cross-validation estimate is only as good as the split behind it. This page shows how each trustcv splitter divides rows into training and test folds, and what goes wrong when the split does not match how the data were collected.

## How splitters divide data

Pick a data structure and a method. Each row of the diagram is one fold (or one fit); each square is one row of data. Training rows are solid, test rows are hatched, and the legend under the diagram names every other state.

<div class="tcv-widget" markdown>

<div class="tcv-controls">
  <div>
    <label for="tcv-split-type">Data structure</label>
    <select id="tcv-split-type">
      <option value="iid">Independent rows</option>
      <option value="grouped">Grouped (patients, sites)</option>
      <option value="temporal">Time-ordered</option>
      <option value="spatial">Spatial</option>
    </select>
  </div>
  <div>
    <label for="tcv-split-method">Method</label>
    <select id="tcv-split-method"></select>
  </div>
  <div>
    <label for="tcv-split-k">Folds: <output id="tcv-split-k-out" for="tcv-split-k">5</output></label>
    <input type="range" id="tcv-split-k" min="2" max="10" value="5">
  </div>
</div>

<div id="tcv-split-viz"></div>
<p id="tcv-split-desc" class="tcv-split-desc" aria-live="polite"></p>

</div>

<small>The diagrams are schematics on small made-up datasets (30 rows, ten patients, or a 12 by 12 map). The fold patterns follow the split logic in `trustcv/splitters/` with the parameters named under each diagram; rows are drawn in fold order where that is easier to read. Run the splitter on your own data to see its exact folds, as in the example below.</small>

What to look for:

| Data structure | The rule the split must keep | Typical splitter |
| --- | --- | --- |
| Independent rows | Test rows are new rows; keep the class balance | `StratifiedKFoldMedical` |
| Grouped | No patient (or site) on both sides | `StratifiedGroupKFold`, `LeaveOneGroupOut` |
| Time-ordered | Never train on rows recorded after the test rows | `TimeSeriesSplit`, `PurgedGroupTimeSeriesSplit` |
| Spatial | Test locations are not surrounded by training neighbours | `SpatialBlockCV`, `BufferedSpatialCV` |

Two methods in the time-ordered list train on data from after the test block: `PurgedKFoldCV` and `CombinatorialPurgedCV`. Purging removes the rows right next to the test block, but the estimate is not forward-only. When the question is "how will this model do on future patients", use a forward-only splitter. `EnvironmentalHealthCV`, which adds environmental variables to spatial and temporal blocks, is not drawn here; see the [API reference](../reference/api.md).

### Print the folds of a real splitter

The same pictures, as text, from the actual splitters. `#` is test, `t` is training and `.` is a row the fold does not use.

```python
import numpy as np
from trustcv import GroupKFoldMedical, PurgedKFoldCV, TimeSeriesSplit

def show(cv, X, **split_args):
    for train, test in cv.split(X, **split_args):
        print("".join("#" if i in test else "t" if i in train else "." for i in range(len(X))))

X = np.zeros((30, 1))
show(TimeSeriesSplit(n_splits=4), X)                 # forward-only, growing training set
print()
show(PurgedKFoldCV(n_splits=4, purge_gap=2), X)      # trains on both sides of the test block
print()
patients = np.repeat(np.arange(10), 3)               # ten patients, three rows each
show(GroupKFoldMedical(n_splits=5, random_state=0), X, groups=patients)
```

## Simulated scenarios

<span class="tcv-sim-badge">Simulated</span> These scenarios are illustrations, not experiments. Pressing **Run simulation** draws five fold scores at random around values chosen to show the usual direction of the effect: a split that leaks scores higher than one that matches the question. No model is trained and no dataset is used, and the size of a real gap depends on your data. For real numbers, run the [notebooks](notebooks.md) or the [tutorial example](tutorials.md#understanding-data-leakage).

<div class="tcv-widget" id="tcv-scenarios" markdown>

<div class="tcv-controls">
  <div>
    <label for="tcv-scn-scenario">Scenario</label>
    <select id="tcv-scn-scenario"></select>
  </div>
  <div>
    <label for="tcv-scn-method">Validation approach</label>
    <select id="tcv-scn-method"></select>
  </div>
  <div>
    <button type="button" id="tcv-scn-run" class="md-button md-button--primary">Run simulation</button>
  </div>
</div>

<p class="tcv-scn-setting"></p>
<div id="tcv-scn-out" aria-live="polite"></div>

</div>

Some approaches in the scenarios are common in their fields but are not trustcv classes: ancestry-stratified, batch-aware, scaffold and target-based splits. They are labelled "concept". You implement them by building a group label (batch, scaffold or target ID) and passing it to a grouped splitter, for example `StratifiedGroupKFold(n_splits=5).split(X, y, groups=scaffold_ids)`.

The related code links point to scripts in the repository: small ones ship with the package in `trustcv/examples/`, heavier ones (deep learning, spatial models) live in `examples/`.

## Next steps

- [Choosing a split](../guides/choosing-cv.md) turns these rules into a decision for your data.
- [Tutorials](tutorials.md) work through each case with code that runs.
- [Quiz](quiz.md) tests the ideas on twenty clinical scenarios.

<script src="../assets/js/cv-visualizations.js"></script>
<script src="../assets/js/cv-temporal.js"></script>
<script src="../assets/js/cv-grouped.js"></script>
<script src="../assets/js/cv-spatial.js"></script>
<script src="../assets/js/tutorials.js"></script>
