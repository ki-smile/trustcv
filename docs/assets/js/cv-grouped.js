/* Grouped splitters for the split visualiser (needs cv-visualizations.js).
 * Thirty visits from ten patients (letters A-J), three hospitals. Logic
 * mirrors trustcv/splitters/grouped.py; each cell shows its patient letter. */
(function () {
  "use strict";
  var S = window.tcvSplits;
  if (!S) return;
  var h = S.helpers;

  var VISITS = [3, 2, 4, 3, 2, 3, 4, 2, 3, 4];          // visits per patient, 30 in total
  var OUTCOME = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0];         // per patient
  var HOSPITAL = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3];        // per patient
  var P = VISITS.length, letters = "ABCDEFGHIJ";
  var patientOf = [];
  VISITS.forEach(function (v, p) { for (var i = 0; i < v; i++) patientOf.push(p); });

  // One row of cells from a per-patient state function.
  function cells(stateOf) {
    return patientOf.map(function (p) { return { s: stateOf(p), t: letters[p] }; });
  }
  function header(label, valueOf) {
    return { label: label, header: true, cells: patientOf.map(function (p) { return { s: "hdr", t: valueOf(p) }; }) };
  }
  // Patients to folds in a shuffled order, sizes as in GroupKFoldMedical.
  function patientFolds(k, seed) {
    var order = h.permutation(P, seed), fold = [];
    h.foldRanges(P, k).forEach(function (r, f) { for (var j = r[0]; j < r[1]; j++) fold[order[j]] = f; });
    return fold;
  }
  function foldRows(fold, k, prefix) {
    var rows = [];
    for (var f = 0; f < k; f++) rows.push({ label: (prefix || "Fold ") + (f + 1), cells: cells(function (p) { return fold[p] === f ? "test" : "train"; }) });
    return rows;
  }

  S.register("grouped", {
    key: "rowlevel", cls: "KFoldMedical(shuffle=True)", label: "Row-level K-fold (wrong for repeated patients)",
    note: "For comparison: splitting visits instead of patients. A dot marks a test visit whose patient also has visits in training, so the model can recognise the patient instead of learning the condition.",
    build: function (k) {
      var p = h.permutation(patientOf.length, 3);
      return { rows: h.foldRanges(patientOf.length, k).map(function (r, f) {
        var isTest = {};
        for (var j = r[0]; j < r[1]; j++) isTest[p[j]] = true;
        var inTrain = {};
        patientOf.forEach(function (pt, i) { if (!isTest[i]) inTrain[pt] = true; });
        return { label: "Fold " + (f + 1), cells: patientOf.map(function (pt, i) {
          return isTest[i] ? { s: "test", t: letters[pt], leak: !!inTrain[pt] } : { s: "train", t: letters[pt] };
        }) };
      }) };
    }
  });
  S.register("grouped", {
    key: "group", cls: "GroupKFoldMedical", label: "Group K-fold", maxK: 10,
    note: "All visits of a patient go to the same fold, so no patient is on both sides. Pass the patient IDs as groups=. Folds can differ in size when patients have different numbers of visits.",
    build: function (k) { return { rows: foldRows(patientFolds(k, 9), k) }; }
  });
  S.register("grouped", {
    key: "sgkf", cls: "StratifiedGroupKFold", label: "Stratified group K-fold", maxK: 10,
    note: "Keeps patients whole and spreads patients with the outcome (+) across folds so each fold has a similar share. Usually the best default for repeated-patient classification.",
    build: function (k) {
      var fold = [], n = 0;
      [1, 0].forEach(function (cls) { OUTCOME.forEach(function (o, p) { if (o === cls) fold[p] = n++ % k; }); });
      return { rows: [header("Outcome", function (p) { return OUTCOME[p] ? "+" : "−"; })].concat(foldRows(fold, k)) };
    }
  });
  S.register("grouped", {
    key: "logo", cls: "LeaveOneGroupOut", label: "Leave one group out", fixedK: true,
    note: "Each patient (or site) is the test set once. Small groups make fold-wise metrics unstable or undefined; trustcv warns when a held-out group has a single class.",
    build: function () {
      var rows = [];
      for (var g = 0; g < P; g++) rows.push({ label: "Out: " + letters[g], cells: cells(function (p) { return p === g ? "test" : "train"; }) });
      return { rows: rows };
    }
  });
  S.register("grouped", {
    key: "lpgo", cls: "LeavePGroupsOut", label: "Leave p groups out (p=2)", fixedK: true,
    note: "Every pair of patients is a test set once, C(10, 2) = 45 splits. The first 8 are shown.",
    build: function () {
      var rows = [];
      for (var a = 0; a < P && rows.length < 8; a++) for (var b = a + 1; b < P && rows.length < 8; b++) {
        rows.push({ label: "Out: " + letters[a] + letters[b], cells: cells(function (p) { return p === a || p === b ? "test" : "train"; }) });
      }
      return { rows: rows };
    }
  });
  S.register("grouped", {
    key: "repeated", cls: "RepeatedGroupKFold", label: "Repeated group K-fold", maxK: 10,
    note: "Group K-fold repeated with a different patient shuffle each time (two repeats shown). Averages out the luck of one assignment.",
    build: function (k) {
      var a = foldRows(patientFolds(k, 9), k, "R1 F"), b = foldRows(patientFolds(k, 21), k, "R2 F");
      return { rows: a.concat(b) };
    }
  });
  S.register("grouped", {
    key: "hier", cls: "HierarchicalGroupKFold", label: "Hierarchical (leave hospital out)", fixedK: true,
    note: "With hierarchy_level=\"hospital\" and the hierarchy passed to split(hierarchy=...), whole hospitals are held out, which tests transfer to a new site. Three hospitals give three folds here.",
    build: function () {
      var rows = [header("Hospital", function (p) { return String(HOSPITAL[p]); })];
      for (var s = 1; s <= 3; s++) rows.push({ label: "Out: H" + s, cells: cells(function (p) { return HOSPITAL[p] === s ? "test" : "train"; }) });
      return { rows: rows };
    }
  });
  S.register("grouped", {
    key: "nested", cls: "NestedGroupedCV", label: "Nested grouped CV", maxK: 5,
    note: "An outer group K-fold estimates performance; an inner group K-fold over the outer training patients (first inner split shown) tunes hyperparameters. Patients stay whole at both levels.",
    build: function (k) {
      var fold = patientFolds(k, 9), rows = [];
      for (var f = 0; f < k; f++) {
        var trainPts = [];
        for (var p = 0; p < P; p++) if (fold[p] !== f) trainPts.push(p);
        var inner = trainPts.slice(0, Math.ceil(trainPts.length / 3));
        rows.push({ label: "Outer " + (f + 1), cells: cells(function (p) { return fold[p] === f ? "test" : inner.indexOf(p) >= 0 ? "val" : "train"; }) });
      }
      return { rows: rows };
    }
  });
})();
