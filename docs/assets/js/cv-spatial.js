/* Spatial splitters for the split visualiser (needs cv-visualizations.js).
 * Each fold is drawn as a 12 x 12 map of sampling locations. Logic mirrors
 * trustcv/splitters/spatial.py with block_shape="grid". */
(function () {
  "use strict";
  var S = window.tcvSplits;
  if (!S) return;
  var h = S.helpers, G = 12;

  // Grid blocks: ceil(sqrt(n_splits)) per side; if that gives more blocks than
  // n_splits, n_splits of them are chosen at random as test blocks.
  function blocks(k) {
    var side = Math.ceil(Math.sqrt(k)), id = [];
    for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
      id.push(Math.min(side - 1, Math.floor(y * side / G)) * side + Math.min(side - 1, Math.floor(x * side / G)));
    }
    var chosen = h.permutation(side * side, 17).slice(0, k).sort(function (a, b) { return a - b; });
    return { id: id, chosen: chosen, side: side };
  }

  S.register("spatial", {
    key: "block", cls: "SpatialBlockCV", label: "Spatial block",
    note: "The map is cut into a grid of blocks and one block is the test set per fold, so test locations are not surrounded by training neighbours. Pass coordinates to split(coordinates=...).",
    build: function (k) {
      var b = blocks(k);
      return { cols: G, extra: b.side * b.side > k ? "With " + k + " folds the grid has " + b.side * b.side + " blocks; " + (b.side * b.side - k) + " are never a test block." : "", rows: b.chosen.map(function (blk, f) {
        return { label: "Fold " + (f + 1), cells: b.id.map(function (v) { return v === blk ? "test" : "train"; }) };
      }) };
    }
  });
  S.register("spatial", {
    key: "buffered", cls: "BufferedSpatialCV", label: "Buffered spatial",
    note: "As spatial block, but locations within buffer_size of the test block are dropped from training. buffer_size is in coordinate units (here 1.5 grid cells); a value below 1 is read as a fraction of the largest distance.",
    build: function (k) {
      var b = blocks(k);
      return { cols: G, rows: b.chosen.map(function (blk, f) {
        var test = [];
        b.id.forEach(function (v, i) { if (v === blk) test.push(i); });
        return { label: "Fold " + (f + 1), cells: b.id.map(function (v, i) {
          if (v === blk) return "test";
          var x = i % G, y = Math.floor(i / G);
          for (var t = 0; t < test.length; t++) {
            var dx = x - test[t] % G, dy = y - Math.floor(test[t] / G);
            if (dx * dx + dy * dy <= 2.25) return "gap";
          }
          return "train";
        }) };
      }) };
    }
  });
  S.register("spatial", {
    key: "st", cls: "SpatiotemporalBlockCV", label: "Spatiotemporal block", fixedK: true,
    note: "Rows are space blocks (n_spatial_blocks=4), columns are time (n_temporal_blocks=3, four time steps each). Every space-time block is the test set once: 12 folds. buffer_space and buffer_time can also exclude neighbours; they are 0 here.",
    build: function () {
      var rows = [];
      for (var s = 0; s < 4; s++) for (var t = 0; t < 3; t++) {
        var c = [];
        for (var r = 0; r < 4; r++) for (var col = 0; col < 12; col++) c.push(r === s && Math.floor(col / 4) === t ? "test" : "train");
        rows.push({ label: "S" + (s + 1) + " T" + (t + 1), cells: c });
      }
      return { cols: 12, rows: rows };
    }
  });
})();
