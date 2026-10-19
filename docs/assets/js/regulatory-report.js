/* trustcv report draft tool.
   Turns values the user enters into a neutral validation-summary draft.
   It computes nothing: no intervals, thresholds or conclusions. Runs only in the browser. */
(function () {
  "use strict";

  var form = document.getElementById("tcv-report-form");
  if (!form) return;

  var DISCLAIMER = "trustcv supports documentation of validation choices. It does not make a model or device compliant with FDA, EU MDR or any other regulation.";
  var NOT_PROVIDED = "not provided";
  var METRICS = [
    ["auc", "AUC-ROC"],
    ["sens", "Sensitivity"],
    ["spec", "Specificity"],
    ["ppv", "Positive predictive value"],
    ["npv", "Negative predictive value"],
    ["acc", "Accuracy"]
  ];
  var CI_METHODS = {
    corrected_t: "corrected_t (Nadeau–Bengio corrected t-interval over fold scores)",
    oof_bootstrap: "oof_bootstrap (bootstrap of pooled out-of-fold predictions)",
    other: "other (see notes)"
  };

  var statusEl = document.getElementById("tcv-report-status");
  var output = document.getElementById("tcv-report-output");
  var preview = document.getElementById("tcv-report-preview");
  var mdButton = document.getElementById("tcv-download-md");
  var htmlButton = document.getElementById("tcv-download-html");
  var current = null;

  // Performance rows: one estimate and two optional bounds per metric, each with its own label.
  var tbody = document.getElementById("perf-rows");
  METRICS.forEach(function (m) {
    var tr = document.createElement("tr");
    var th = document.createElement("th");
    th.scope = "row";
    th.textContent = m[1];
    tr.appendChild(th);
    [["est", "estimate"], ["lo", "lower bound"], ["hi", "upper bound"]].forEach(function (part) {
      var td = document.createElement("td");
      var id = "f-" + m[0] + "-" + part[0];
      var label = document.createElement("label");
      label.htmlFor = id;
      label.className = "tcv-sr-only";
      label.textContent = m[1] + " " + part[1];
      var input = document.createElement("input");
      input.id = id;
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      td.appendChild(label);
      td.appendChild(input);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  function field(id) { return document.getElementById(id); }
  function val(id) { return field(id).value.trim(); }
  function or(v) { return v === "" ? NOT_PROVIDED : v; }

  function validate() {
    var errors = [];
    form.querySelectorAll("[aria-invalid]").forEach(function (el) { el.removeAttribute("aria-invalid"); });
    function fail(id, msg) { field(id).setAttribute("aria-invalid", "true"); errors.push({ id: id, msg: msg }); }

    if (!val("f-name")) fail("f-name", "Enter the model or software name.");
    if (!val("f-version")) fail("f-version", "Enter the version.");
    form.querySelectorAll('input[type="number"]').forEach(function (el) {
      if (el.validity.badInput) fail(el.id, document.querySelector('label[for="' + el.id + '"]').textContent + " is not a number.");
    });
    var level = val("f-cilevel");
    if (level !== "" && !(Number(level) > 0 && Number(level) < 100)) fail("f-cilevel", "The interval level must be between 0 and 100.");

    METRICS.forEach(function (m) {
      var est = val("f-" + m[0] + "-est"), lo = val("f-" + m[0] + "-lo"), hi = val("f-" + m[0] + "-hi");
      if ((lo === "") !== (hi === "")) fail("f-" + m[0] + (lo === "" ? "-lo" : "-hi"), m[1] + ": enter both interval bounds or neither.");
      else if (lo !== "" && est === "") fail("f-" + m[0] + "-est", m[1] + ": enter the estimate as well as the interval.");
      else if (lo !== "" && !(Number(lo) <= Number(est) && Number(est) <= Number(hi))) fail("f-" + m[0] + "-lo", m[1] + ": the estimate should lie between the lower and upper bound.");
    });

    var counts = ["f-tp", "f-fp", "f-fn", "f-tn"];
    var filled = counts.filter(function (id) { return val(id) !== ""; });
    if (filled.length > 0 && filled.length < 4) {
      counts.forEach(function (id) { if (val(id) === "") fail(id, "Enter all four confusion matrix counts, or none."); });
    }
    filled.forEach(function (id) {
      var n = Number(val(id));
      if (!(Number.isInteger(n) && n >= 0)) fail(id, "Counts must be whole numbers of zero or more.");
    });
    return errors;
  }

  // The draft as a list of sections; rendered both to Markdown and to HTML.
  function buildDoc() {
    var level = val("f-cilevel");  // "" means the user did not state it; never assume 95%
    var doc = { title: "Validation summary (draft): " + val("f-name") + " " + val("f-version"), sections: [],
      file: "validation-summary-" + slug(val("f-name") + "-" + val("f-version")) };
    function section(title, blocks) { doc.sections.push({ title: title, blocks: blocks }); }
    function para(text) { return { p: text }; }
    function textOr(id) { return para(val(id) || "Not provided."); }

    doc.intro = [
      para(DISCLAIMER),
      para("Draft prepared on " + new Date().toISOString().slice(0, 10) + " with the trustcv report draft tool. Every value below was entered by the author; the tool did not compute or check any of them. Check each value against the analysis output before using this document.")
    ];

    section("Model and intended use", [
      { table: [["Item", "Value"], ["Name", val("f-name")], ["Version", val("f-version")], ["Developer or manufacturer", or(val("f-developer"))]] },
      para("Intended use: " + (val("f-use") || NOT_PROVIDED))
    ]);
    section("Data", [
      { table: [["Item", "Value"], ["Patients", or(val("f-patients"))], ["Rows, images or samples", or(val("f-rows"))]] },
      para("Sources, sites and period: " + (val("f-sources") || NOT_PROVIDED)),
      para("Population: " + (val("f-population") || NOT_PROVIDED))
    ]);
    section("Validation design", [
      { table: [["Item", "Value"],
        ["Split method", or(val("f-method"))],
        ["Number of folds", or(val("f-folds"))],
        ["Grouping unit", or(val("f-group"))],
        ["Random seed", or(val("f-seed"))],
        ["Interval method", CI_METHODS[val("f-cimethod")] || NOT_PROVIDED],
        ["Interval level", level ? level + "%" : "not provided"],
        ["trustcv overall status", or(val("f-status"))],
        ["trustcv version", or(val("f-tcvversion"))]] },
      para("Why this split suits the data: " + (val("f-justification") || NOT_PROVIDED)),
      para("Leakage controls and integrity check notes: " + (val("f-leakage") || NOT_PROVIDED))
    ]);

    var rows = [["Metric", "Estimate", level ? level + "% confidence interval" : "Confidence interval (level not provided)"]];
    METRICS.forEach(function (m) {
      var est = val("f-" + m[0] + "-est"), lo = val("f-" + m[0] + "-lo"), hi = val("f-" + m[0] + "-hi");
      if (est !== "") rows.push([m[1], est, lo !== "" ? lo + " to " + hi : NOT_PROVIDED]);
    });
    section("Performance", rows.length > 1 ? [{ table: rows }] : [para("No performance values were entered.")]);

    if (["f-tp", "f-fp", "f-fn", "f-tn"].every(function (id) { return val(id) !== ""; })) {
      var tp = Number(val("f-tp")), fp = Number(val("f-fp")), fn = Number(val("f-fn")), tn = Number(val("f-tn"));
      section("Confusion matrix", [
        { table: [["", "Predicted positive", "Predicted negative"], ["Actual positive", String(tp), String(fn)], ["Actual negative", String(fp), String(tn)]] },
        para("Counts as entered; total " + (tp + fp + fn + tn) + ". Decision threshold: " + (val("f-threshold") || NOT_PROVIDED) + ".")
      ]);
    }

    var subgroups = val("f-subgroups").split("\n").map(function (s) { return s.trim(); }).filter(Boolean);
    section("Subgroup results", [subgroups.length ? { list: subgroups } : para("Not provided.")]);
    section("Limitations", [textOr("f-limitations")]);
    if (val("f-notes")) section("Other notes", [textOr("f-notes")]);
    return doc;
  }

  function mdCell(s) { return String(s).replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " "); }
  function toMarkdown(doc) {
    var out = ["# " + doc.title, ""];
    function blocks(list) {
      list.forEach(function (b) {
        if (b.p !== undefined) out.push(b.p, "");
        else if (b.list) { b.list.forEach(function (i) { out.push("- " + i); }); out.push(""); }
        else if (b.table) {
          b.table.forEach(function (r, i) {
            out.push("| " + r.map(mdCell).join(" | ") + " |");
            if (i === 0) out.push("|" + r.map(function () { return " --- "; }).join("|") + "|");
          });
          out.push("");
        }
      });
    }
    blocks(doc.intro);
    doc.sections.forEach(function (s, i) { out.push("## " + (i + 1) + ". " + s.title, ""); blocks(s.blocks); });
    return out.join("\n");
  }

  // DOM rendering uses textContent only, so entered text is never parsed as HTML.
  function el(tag, text) { var e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; }
  function renderBlocks(parent, list, headingTag) {
    list.forEach(function (b) {
      if (b.p !== undefined) parent.appendChild(el("p", b.p));
      else if (b.list) { var ul = el("ul"); b.list.forEach(function (i) { ul.appendChild(el("li", i)); }); parent.appendChild(ul); }
      else if (b.table) {
        var t = el("table"), thead = el("thead"), tb = el("tbody");
        b.table.forEach(function (r, i) {
          var tr = el("tr");
          r.forEach(function (c, j) {
            var cell = el(i === 0 || j === 0 ? "th" : "td", c);
            if (i === 0) cell.scope = "col"; else if (j === 0) cell.scope = "row";
            tr.appendChild(cell);
          });
          (i === 0 ? thead : tb).appendChild(tr);
        });
        t.appendChild(thead); t.appendChild(tb);
        var wrap = el("div"); wrap.className = "tcv-table-wrap"; wrap.appendChild(t);
        parent.appendChild(wrap);
      }
    });
  }
  function toNode(doc, h1, h2) {
    var root = el("div");
    root.appendChild(el(h1, doc.title));
    renderBlocks(root, doc.intro);
    doc.sections.forEach(function (s, i) { root.appendChild(el(h2, (i + 1) + ". " + s.title)); renderBlocks(root, s.blocks); });
    return root;
  }
  function toHtmlFile(doc) {
    var body = toNode(doc, "h1", "h2").innerHTML;
    var title = el("title", doc.title).outerHTML;
    return "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" + title +
      "\n<style>body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a;background:#fff}" +
      "table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #888;padding:.3rem .6rem;text-align:left;vertical-align:top}</style>\n</head>\n<body>\n" +
      body + "\n</body>\n</html>\n";
  }

  function slug(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "draft"; }
  function download(text, type, name) {
    var blob = new Blob([text], { type: type + ";charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    statusEl.textContent = "Downloaded " + a.download + ".";
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var errors = validate();
    if (errors.length) {
      current = null;
      mdButton.disabled = htmlButton.disabled = true;
      statusEl.textContent = (errors.length === 1 ? "One field needs attention: " : errors.length + " fields need attention: ") +
        errors.map(function (x) { return x.msg; }).filter(function (m, i, all) { return all.indexOf(m) === i; }).join(" ");
      field(errors[0].id).focus();
      return;
    }
    current = buildDoc();
    preview.replaceChildren(toNode(current, "h3", "h4"));
    output.hidden = false;
    mdButton.disabled = htmlButton.disabled = false;
    statusEl.textContent = "Draft generated below. Review and edit it before use; download it as Markdown or HTML.";
    document.getElementById("tcv-report-output-title").focus();
  });

  // Edits after generation make the preview stale; say so instead of silently diverging.
  form.addEventListener("input", function () {
    if (current) statusEl.textContent = "The form has changed since the draft was generated. Generate it again to update the preview and downloads.";
  });

  mdButton.addEventListener("click", function () { if (current) download(toMarkdown(current), "text/markdown", current.file + ".md"); });
  htmlButton.addEventListener("click", function () { if (current) download(toHtmlFile(current), "text/html", current.file + ".html"); });
})();
