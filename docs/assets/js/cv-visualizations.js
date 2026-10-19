/* Split visualiser for the "Interactive visualisations" page.
 *
 * Draws, for each fold, which rows a trustcv splitter puts in training and
 * which in test. The patterns follow the split logic in trustcv/splitters/;
 * they are schematic (small made-up datasets), not output of a run.
 *
 * This file holds the shared renderer and the independent-sample (IID)
 * methods. cv-temporal.js, cv-grouped.js and cv-spatial.js add their methods
 * through window.tcvSplits.register(). Colours come from CSS classes in
 * learn.css that use the trustcv tokens, so both colour schemes work.
 */
(function () {
  "use strict";

  var methods = { iid: [], temporal: [], grouped: [], spatial: [] };

  // Small seeded PRNG so diagrams are stable between renders.
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function permutation(n, seed) {
    var r = rng(seed), p = [];
    for (var i = 0; i < n; i++) p.push(i);
    for (var j = n - 1; j > 0; j--) {
      var k = Math.floor(r() * (j + 1));
      var tmp = p[j]; p[j] = p[k]; p[k] = tmp;
    }
    return p;
  }
  function fill(n, state) {
    var a = [];
    for (var i = 0; i < n; i++) a.push(state);
    return a;
  }
  // Contiguous K-fold boundaries (sklearn/trustcv style: first n % k folds get one extra).
  function foldRanges(n, k) {
    var out = [], start = 0;
    for (var f = 0; f < k; f++) {
      var size = Math.floor(n / k) + (f < n % k ? 1 : 0);
      out.push([start, start + size]);
      start += size;
    }
    return out;
  }

  var helpers = { rng: rng, permutation: permutation, fill: fill, foldRanges: foldRanges };

  function register(type, def) { methods[type].push(def); }

  /* ---------- IID methods ---------- */
  var N = 30;
  // Outcome labels for the stratified view: every third row is positive.
  var outcome = [];
  for (var i = 0; i < N; i++) outcome.push(i % 3 === 1 ? 1 : 0);
  function outcomeRow(labels) {
    return { label: "Outcome", header: true, cells: labels.map(function (v) { return { s: "hdr", t: v ? "+" : "−" }; }) };
  }

  register("iid", {
    key: "holdout", cls: "HoldOut", label: "Hold-out (single split)", fixedK: true,
    note: "One random split, here test_size=0.2. Fast, but the estimate depends on which rows happened to land in test.",
    build: function () {
      var cells = fill(N, "train");
      permutation(N, 7).slice(0, Math.round(N * 0.2)).forEach(function (j) { cells[j] = "test"; });
      return { rows: [{ label: "Split", cells: cells }] };
    }
  });
  register("iid", {
    key: "kfold", cls: "KFoldMedical", label: "K-fold",
    note: "The rows are cut into K folds; each fold is the test set once. KFoldMedical does not shuffle by default, so folds are contiguous.",
    build: function (k) {
      return { rows: foldRanges(N, k).map(function (r, f) {
        var c = fill(N, "train");
        for (var j = r[0]; j < r[1]; j++) c[j] = "test";
        return { label: "Fold " + (f + 1), cells: c };
      }) };
    }
  });
  register("iid", {
    key: "stratified", cls: "StratifiedKFoldMedical", label: "Stratified K-fold",
    note: "Like K-fold, but each test fold keeps the overall share of positive cases (the + row). Use it for imbalanced outcomes.",
    build: function (k) {
      var rank = [0, 0], pos = [];
      outcome.forEach(function (v) { pos.push(rank[v]++); });
      var rows = [outcomeRow(outcome)];
      for (var f = 0; f < k; f++) {
        rows.push({ label: "Fold " + (f + 1), cells: outcome.map(function (v, j) { return pos[j] % k === f ? "test" : "train"; }) });
      }
      return { rows: rows };
    }
  });
  register("iid", {
    key: "repeated", cls: "RepeatedKFold", label: "Repeated K-fold",
    note: "K-fold run several times with a different shuffle each time (two repeats shown; the default is n_repeats=10).",
    build: function (k) {
      var rows = [];
      [11, 23].forEach(function (seed, rep) {
        var p = permutation(N, seed);
        foldRanges(N, k).forEach(function (r, f) {
          var c = fill(N, "train");
          for (var j = r[0]; j < r[1]; j++) c[p[j]] = "test";
          rows.push({ label: "R" + (rep + 1) + " F" + (f + 1), cells: c });
        });
      });
      return { rows: rows };
    }
  });
  register("iid", {
    key: "loocv", cls: "LOOCV", label: "Leave-one-out", fixedK: true,
    note: "Each row is the test set once, so there are as many fits as rows (10 rows shown). Low bias, but the estimate has high variance and it is expensive.",
    build: function () {
      var rows = [];
      for (var f = 0; f < 10; f++) { var c = fill(10, "train"); c[f] = "test"; rows.push({ label: "Fit " + (f + 1), cells: c }); }
      return { rows: rows };
    }
  });
  register("iid", {
    key: "lpocv", cls: "LPOCV", label: "Leave-p-out (p=2)", fixedK: true,
    note: "Every pair of rows is a test set once: C(n, 2) fits. With 6 rows that is 15; it grows very quickly with n.",
    build: function () {
      var rows = [], n = 6;
      for (var a = 0; a < n; a++) for (var b = a + 1; b < n; b++) {
        var c = fill(n, "train"); c[a] = "test"; c[b] = "test";
        rows.push({ label: "Fit " + (rows.length + 1), cells: c });
      }
      return { rows: rows };
    }
  });
  register("iid", {
    key: "bootstrap", cls: "BootstrapValidation", label: "Bootstrap (.632)", fixedK: true,
    note: "Each iteration draws n rows with replacement for training (a number marks rows drawn more than once). Rows never drawn, about 36.8% on average, are the out-of-bag test set. Five of the default 100 iterations shown.",
    build: function () {
      var rows = [], n = 20;
      for (var it = 0; it < 5; it++) {
        var r = rng(100 + it), count = fill(n, 0);
        for (var d = 0; d < n; d++) count[Math.floor(r() * n)]++;
        rows.push({ label: "Iter " + (it + 1), cells: count.map(function (c) { return c ? { s: "train", t: c > 1 ? String(c) : "" } : "test"; }) });
      }
      return { rows: rows };
    }
  });
  register("iid", {
    key: "montecarlo", cls: "MonteCarloCV", label: "Monte Carlo (repeated random splits)",
    note: "Each iteration draws a fresh random 20% test set, so test sets can overlap between iterations. The slider sets the number of iterations shown.",
    build: function (k) {
      var rows = [];
      for (var it = 0; it < k; it++) {
        var c = fill(N, "train");
        permutation(N, 300 + it).slice(0, 6).forEach(function (j) { c[j] = "test"; });
        rows.push({ label: "Iter " + (it + 1), cells: c });
      }
      return { rows: rows };
    }
  });
  register("iid", {
    key: "nested", cls: "NestedCV", label: "Nested CV",
    note: "The outer K-fold estimates performance. Inside each outer training set an inner K-fold (3 folds; the first inner split shown) picks hyperparameters, so the outer test fold is never used for tuning.",
    build: function (k) {
      return { rows: foldRanges(N, k).map(function (r, f) {
        var c = fill(N, "train"), trainIdx = [];
        for (var j = 0; j < N; j++) { if (j >= r[0] && j < r[1]) c[j] = "test"; else trainIdx.push(j); }
        var inner = foldRanges(trainIdx.length, 3)[0];
        for (var q = inner[0]; q < inner[1]; q++) c[trainIdx[q]] = "val";
        return { label: "Outer " + (f + 1), cells: c };
      }) };
    }
  });

  /* ---------- Renderer ---------- */
  var SVGNS = "http://www.w3.org/2000/svg";
  var STATE_TEXT = {
    train: "Training",
    test: "Test",
    val: "Inner validation (tuning only)",
    gap: "Excluded (purge, embargo or buffer)",
    unused: "Not used in this split",
    leak: "Shares a patient with the other side (leak)"
  };
  var uid = 0;

  function el(name, attrs, parent) {
    var e = document.createElementNS(SVGNS, name);
    for (var a in attrs) e.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function norm(cell) { return typeof cell === "string" ? { s: cell } : cell; }

  function defs(svg) {
    var id = "tcv" + (++uid), d = el("defs", {}, svg);
    var p = el("pattern", { id: id + "h", width: 4, height: 4, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, d);
    el("rect", { width: 1.6, height: 4, "class": "tcv-hatch" }, p);
    var q = el("pattern", { id: id + "d", width: 4, height: 4, patternUnits: "userSpaceOnUse" }, d);
    el("circle", { cx: 2, cy: 2, r: 0.9, "class": "tcv-hatch" }, q);
    return id;
  }

  function drawCell(svg, id, x, y, size, cell) {
    var c = norm(cell);
    el("rect", { x: x, y: y, width: size, height: size, rx: 1.5, "class": "tcv-c tcv-c-" + c.s }, svg);
    if (c.s === "test") el("rect", { x: x, y: y, width: size, height: size, rx: 1.5, fill: "url(#" + id + "h)" }, svg);
    if (c.s === "val") el("rect", { x: x, y: y, width: size, height: size, rx: 1.5, fill: "url(#" + id + "d)" }, svg);
    if (c.leak) el("circle", { cx: x + size / 2, cy: y + size / 2, r: size * 0.22, "class": "tcv-leak" }, svg);
    if (c.t) {
      var t = el("text", { x: x + size / 2, y: y + size / 2, "class": "tcv-ctext" + (c.s === "hdr" ? " tcv-ctext-hdr" : ""), "text-anchor": "middle", "dominant-baseline": "central", "font-size": size * 0.62 }, svg);
      t.textContent = c.t;
    }
  }

  function summarise(row) {
    var n = {};
    row.cells.forEach(function (cell) { var c = norm(cell); n[c.s] = (n[c.s] || 0) + 1; if (c.leak) n.leak = (n.leak || 0) + 1; });
    var parts = [];
    ["train", "test", "val", "gap", "unused", "leak"].forEach(function (s) { if (n[s]) parts.push(n[s] + " " + STATE_TEXT[s].split(" (")[0].toLowerCase()); });
    return row.label + ": " + parts.join(", ");
  }

  // Strip layout: one row of cells per fold, time or index left to right.
  function drawStrips(fig, spec) {
    var size = 12, gap = 2, labelW = 58, n = spec.rows[0].cells.length;
    var W = labelW + n * (size + gap), H = spec.rows.length * (size + gap + 2);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "class": "tcv-split-svg" });
    svg.style.maxWidth = Math.max(W * 1.25, 260) + "px";
    var id = defs(svg);
    spec.rows.forEach(function (row, r) {
      var y = r * (size + gap + 2);
      var lab = el("text", { x: 0, y: y + size / 2, "dominant-baseline": "central", "font-size": 9, "class": "tcv-rowlabel" }, svg);
      lab.textContent = row.label;
      row.cells.forEach(function (cell, j) { drawCell(svg, id, labelW + j * (size + gap), y, size, cell); });
    });
    return svg;
  }

  // Map layout: each fold is a small grid (spatial methods).
  function drawMaps(fig, spec) {
    var wrap = document.createElement("div");
    wrap.className = "tcv-maps";
    var cols = spec.cols, size = 9, gap = 1;
    spec.rows.forEach(function (row) {
      var rowsN = Math.ceil(row.cells.length / cols);
      var W = cols * (size + gap), H = rowsN * (size + gap) + 14;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "class": "tcv-map-svg" });
      var id = defs(svg);
      var lab = el("text", { x: 0, y: 6, "dominant-baseline": "central", "font-size": 9, "class": "tcv-rowlabel" }, svg);
      lab.textContent = row.label;
      row.cells.forEach(function (cell, j) {
        drawCell(svg, id, (j % cols) * (size + gap), 14 + Math.floor(j / cols) * (size + gap), size, cell);
      });
      svg.setAttribute("aria-label", summarise(row));
      wrap.appendChild(svg);
    });
    return wrap;
  }

  function legend(spec) {
    var seen = {};
    spec.rows.forEach(function (row) {
      if (row.header) return;
      row.cells.forEach(function (cell) { var c = norm(cell); seen[c.s] = true; if (c.leak) seen.leak = true; });
    });
    var ul = document.createElement("ul");
    ul.className = "tcv-legend";
    ["train", "test", "val", "gap", "unused", "leak"].forEach(function (s) {
      if (!seen[s]) return;
      var li = document.createElement("li");
      var svg = el("svg", { viewBox: "0 0 12 12", width: 14, height: 14, "aria-hidden": "true" });
      var id = defs(svg);
      drawCell(svg, id, 0, 0, 12, s === "leak" ? { s: "train", leak: true } : s);
      li.appendChild(svg);
      li.appendChild(document.createTextNode(" " + STATE_TEXT[s]));
      ul.appendChild(li);
    });
    return ul;
  }

  function init() {
    var root = document.getElementById("tcv-split-viz");
    if (!root) return;
    var typeSel = document.getElementById("tcv-split-type");
    var methodSel = document.getElementById("tcv-split-method");
    var kInput = document.getElementById("tcv-split-k");
    var kOut = document.getElementById("tcv-split-k-out");
    var desc = document.getElementById("tcv-split-desc");
    if (!typeSel || !methodSel || !kInput) return;

    function fillMethods() {
      methodSel.innerHTML = "";
      methods[typeSel.value].forEach(function (m) {
        var o = document.createElement("option");
        o.value = m.key; o.textContent = m.label;
        methodSel.appendChild(o);
      });
    }
    function current() {
      var list = methods[typeSel.value];
      for (var i = 0; i < list.length; i++) if (list[i].key === methodSel.value) return list[i];
      return list[0];
    }
    function render() {
      var m = current();
      if (!m) return;
      var k = parseInt(kInput.value, 10) || 5;
      if (m.minK && k < m.minK) k = m.minK;
      if (m.maxK && k > m.maxK) k = m.maxK;
      kInput.disabled = !!m.fixedK;
      if (kOut) kOut.textContent = m.fixedK ? "set by the method" : String(k);
      var spec = m.build(k);
      root.innerHTML = "";
      var drawing = spec.cols ? drawMaps(root, spec) : drawStrips(root, spec);
      var summary = spec.rows.filter(function (r) { return !r.header; }).map(summarise);
      var label = m.cls + " schematic. " + (summary.length > 8 ? summary.slice(0, 8).join("; ") + "; and " + (summary.length - 8) + " more" : summary.join("; "));
      if (drawing.tagName.toLowerCase() === "svg") drawing.setAttribute("aria-label", label);
      else { drawing.setAttribute("role", "group"); drawing.setAttribute("aria-label", m.cls + " schematic, one map per fold"); }
      var scroller = document.createElement("div");
      scroller.className = "tcv-split-scroll";
      scroller.appendChild(drawing);
      root.appendChild(scroller);
      root.appendChild(legend(spec));
      if (desc) {
        desc.innerHTML = "";
        var code = document.createElement("code");
        code.textContent = m.cls;
        desc.appendChild(code);
        desc.appendChild(document.createTextNode(": " + m.note + (spec.extra ? " " + spec.extra : "")));
      }
    }

    typeSel.addEventListener("change", function () { fillMethods(); render(); });
    methodSel.addEventListener("change", render);
    kInput.addEventListener("input", render);
    fillMethods();
    render();
    // Colours follow CSS tokens; re-render anyway when Material switches scheme.
    new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  window.tcvSplits = { register: register, helpers: helpers, methods: methods, N: N };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else window.addEventListener("load", init);
})();
