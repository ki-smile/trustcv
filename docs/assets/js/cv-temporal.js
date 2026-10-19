/* Temporal splitters for the split visualiser (needs cv-visualizations.js).
 * Rows are in time order, oldest on the left. Logic mirrors
 * trustcv/splitters/temporal.py with the parameters named in each note. */
(function () {
  "use strict";
  var S = window.tcvSplits;
  if (!S) return;
  var h = S.helpers, N = S.N, fill = h.fill;

  function range(c, a, b, state) { for (var j = Math.max(0, a); j < Math.min(c.length, b); j++) c[j] = state; return c; }
  function row(label, c) { return { label: label, cells: c }; }

  S.register("temporal", {
    key: "tss", cls: "TimeSeriesSplit", label: "Time series split",
    note: "Train on everything before the test window, test on the next window. Each test window has n // (n_splits + 1) rows; gap= can leave rows out between them.",
    build: function (k) {
      var nt = Math.floor(N / (k + 1)), rows = [];
      for (var i = 0; i < k; i++) {
        var ts = N - (k - i) * nt, c = fill(N, "unused");
        range(c, 0, ts, "train"); range(c, ts, ts + nt, "test");
        rows.push(row("Fold " + (i + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "blocked", cls: "BlockedTimeSeries", label: "Blocked time series", maxK: 14,
    note: "Rows are grouped into blocks (here block_size=2) that are never split; training blocks always come before test blocks.",
    build: function (k) {
      var nb = N / 2, ts = Math.floor(nb / (k + 1)), rows = [];
      for (var i = 0; i < k; i++) {
        var c = fill(N, "unused");
        range(c, 0, (i + 1) * ts * 2, "train"); range(c, (i + 1) * ts * 2, Math.min((i + 2) * ts, nb) * 2, "test");
        rows.push(row("Fold " + (i + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "rolling", cls: "RollingWindowCV", label: "Rolling window", fixedK: true,
    note: "A fixed-size training window (window_size=10) slides forward by step_size=4 and predicts the next forecast_horizon=4 rows. Old data drops out.",
    build: function () {
      var rows = [];
      for (var s = 0; s + 14 <= N; s += 4) rows.push(row("Fold " + (rows.length + 1), range(range(fill(N, "unused"), s, s + 10, "train"), s + 10, s + 14, "test")));
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "expanding", cls: "ExpandingWindowCV", label: "Expanding window", fixedK: true,
    note: "Training always starts at the first row and grows (initial_train_size=10, step_size=4); each split predicts the next forecast_horizon=4 rows.",
    build: function () {
      var rows = [];
      for (var e = 10; e + 4 <= N; e += 4) rows.push(row("Fold " + (rows.length + 1), range(range(fill(N, "unused"), 0, e, "train"), e, e + 4, "test")));
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "purged", cls: "PurgedKFoldCV", label: "Purged K-fold",
    note: "Each block is the test set once, and purge_gap=2 rows on both sides of it are dropped from training. Training still includes rows from after the test block, so this is not a forward-only estimate; for a deployment-like estimate use a forward-only splitter such as TimeSeriesSplit or PurgedGroupTimeSeriesSplit.",
    build: function (k) {
      var fs = Math.floor(N / k), rows = [];
      for (var i = 0; i < k; i++) {
        var a = i * fs, b = i < k - 1 ? a + fs : N, c = fill(N, "train");
        range(c, a - 2, a, "gap"); range(c, b, b + 2, "gap"); range(c, a, b, "test");
        rows.push(row("Fold " + (i + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "cpcv", cls: "CombinatorialPurgedCV", label: "Combinatorial purged", minK: 3,
    note: "The series is cut into n_splits groups and n_test_splits=2 groups form each test set, with purge_gap=1 around them. With the default strict_order=True only one combination per starting group is kept, so the slider gives n_splits - 1 splits. Like purged K-fold, training includes later rows.",
    build: function (k) {
      var gs = Math.floor(N / k), rows = [];
      function bounds(g) { return [g * gs, g < k - 1 ? (g + 1) * gs : N]; }
      for (var g = 0; g < k - 1; g++) {
        var c = fill(N, "train");
        [g, g + 1].forEach(function (t) { var b = bounds(t); range(c, b[0] - 1, b[0], "gap"); range(c, b[1], b[1] + 1, "gap"); });
        [g, g + 1].forEach(function (t) { var b = bounds(t); range(c, b[0], b[1], "test"); });
        rows.push(row("Split " + (g + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "pgts", cls: "PurgedGroupTimeSeriesSplit", label: "Purged group time series",
    note: "Forward-only: training is everything before the test window minus a purge_gap=2 rows. With groups= and group_exclusive=True, patients in the test window are also removed from training.",
    build: function (k) {
      var nt = Math.floor(N / (k + 1)), rows = [];
      for (var i = 0; i < k; i++) {
        var a = (i + 1) * nt, c = fill(N, "unused");
        range(c, 0, a - 2, "train"); range(c, a - 2, a, "gap"); range(c, a, Math.min(a + nt, N), "test");
        rows.push(row("Fold " + (i + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "nested_temporal", cls: "NestedTemporalCV", label: "Nested temporal",
    note: "An outer forward-only split (TimeSeriesSplit here) estimates performance; an inner forward-only split inside each outer training window tunes hyperparameters (the last inner split shown).",
    build: function (k) {
      var nt = Math.floor(N / (k + 1)), rows = [];
      for (var i = 0; i < k; i++) {
        var ts = N - (k - i) * nt, c = fill(N, "unused"), inner = Math.floor(ts / 4);
        range(c, 0, ts - inner, "train"); range(c, ts - inner, ts, "val"); range(c, ts, ts + nt, "test");
        rows.push(row("Outer " + (i + 1), c));
      }
      return { rows: rows };
    }
  });
  S.register("temporal", {
    key: "shuffled", cls: "KFoldMedical(shuffle=True)", label: "Shuffled K-fold (wrong for time series)",
    note: "For comparison: shuffled K-fold scatters test rows through time, so every fold trains on rows recorded after the ones it is tested on. The model can use the future.",
    build: function (k) {
      var p = h.permutation(N, 5);
      return { rows: h.foldRanges(N, k).map(function (r, f) {
        var c = fill(N, "train");
        for (var j = r[0]; j < r[1]; j++) c[p[j]] = "test";
        return row("Fold " + (f + 1), c);
      }) };
    }
  });
})();
