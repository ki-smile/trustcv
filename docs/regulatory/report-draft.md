---
description: Draft a neutral validation-summary document in the browser from values you enter. Nothing is computed, judged or sent anywhere.
---

# Report draft tool

<link rel="stylesheet" href="../assets/css/regulatory.css">

<div class="tcv-disclaimer" markdown>trustcv supports documentation of validation choices. It does not make a model or device compliant with FDA, EU MDR or any other regulation.</div>

This form turns values you enter into a draft validation summary that you then edit. It runs entirely in your browser; nothing is uploaded.

The tool computes nothing. Point estimates and confidence intervals appear exactly as you enter them, and an interval left blank is shown as "not provided". It applies no thresholds and draws no conclusions. Take the numbers from your own analysis, for example `result.summary()` or the record described in [Documenting validation](documentation-guide.md).

<form id="tcv-report-form" class="tcv-form" novalidate>

<fieldset>
<legend>Model and intended use</legend>
<div class="tcv-grid">
<div class="tcv-field"><label for="f-name">Model or software name (required)</label><input id="f-name" name="name" type="text" required aria-required="true" autocomplete="off"></div>
<div class="tcv-field"><label for="f-version">Version (required)</label><input id="f-version" name="version" type="text" required aria-required="true" autocomplete="off"></div>
<div class="tcv-field"><label for="f-developer">Developer or manufacturer</label><input id="f-developer" name="developer" type="text" autocomplete="organization"></div>
</div>
<div class="tcv-field"><label for="f-use">Intended use, as you would state it</label><textarea id="f-use" name="use" rows="3"></textarea></div>
</fieldset>

<fieldset>
<legend>Data</legend>
<div class="tcv-grid">
<div class="tcv-field"><label for="f-patients">Number of patients</label><input id="f-patients" name="patients" type="number" min="0" step="1" inputmode="numeric"></div>
<div class="tcv-field"><label for="f-rows">Number of rows, images or samples</label><input id="f-rows" name="rows" type="number" min="0" step="1" inputmode="numeric"></div>
</div>
<div class="tcv-field"><label for="f-sources">Data sources, sites and collection period</label><textarea id="f-sources" name="sources" rows="2"></textarea></div>
<div class="tcv-field"><label for="f-population">Population characteristics</label><textarea id="f-population" name="population" rows="2"></textarea></div>
</fieldset>

<fieldset>
<legend>Validation design</legend>
<div class="tcv-grid">
<div class="tcv-field"><label for="f-method">Split method</label><input id="f-method" name="method" type="text" placeholder="stratified_group_kfold" autocomplete="off"></div>
<div class="tcv-field"><label for="f-folds">Number of folds</label><input id="f-folds" name="folds" type="number" min="1" step="1" inputmode="numeric"></div>
<div class="tcv-field"><label for="f-group">Grouping unit</label><input id="f-group" name="group" type="text" placeholder="patient" autocomplete="off"></div>
<div class="tcv-field"><label for="f-seed">Random seed</label><input id="f-seed" name="seed" type="text" autocomplete="off"></div>
<div class="tcv-field"><label for="f-cimethod">Interval method</label>
<select id="f-cimethod" name="cimethod">
<option value="">Not stated</option>
<option value="corrected_t">corrected_t (corrected t-interval over folds)</option>
<option value="oof_bootstrap">oof_bootstrap (bootstrap of out-of-fold predictions)</option>
<option value="other">Other (describe under notes)</option>
</select></div>
<div class="tcv-field"><label for="f-cilevel">Interval level (%)</label><input id="f-cilevel" name="cilevel" type="number" min="1" max="99.9" step="any" value="95" inputmode="decimal"></div>
<div class="tcv-field"><label for="f-status">trustcv overall status</label>
<select id="f-status" name="status">
<option value="">Not reported</option>
<option>PASSED</option>
<option>NEEDS_REVIEW</option>
<option>NOT_FULLY_VERIFIED</option>
<option>FAILED</option>
</select></div>
<div class="tcv-field"><label for="f-tcvversion">trustcv version</label><input id="f-tcvversion" name="tcvversion" type="text" value="1.2.1" autocomplete="off"></div>
</div>
<div class="tcv-field"><label for="f-justification">Why this split suits the data</label><textarea id="f-justification" name="justification" rows="2"></textarea></div>
<div class="tcv-field"><label for="f-leakage">Leakage controls and integrity check notes</label><textarea id="f-leakage" name="leakage" rows="2" aria-describedby="f-leakage-hint"></textarea><p id="f-leakage-hint" class="tcv-hint">For example: preprocessing inside the pipeline, tuning in nested CV, checks that were not run and why.</p></div>
</fieldset>

<fieldset>
<legend>Performance</legend>
<p class="tcv-hint" id="perf-hint">Enter values on the scale you report them (for example 0.87 or 87). Leave a row empty to omit it. Interval bounds are optional, but enter both or neither.</p>
<div class="tcv-table-wrap">
<table class="tcv-perf" aria-describedby="perf-hint">
<thead><tr><th scope="col">Metric</th><th scope="col">Estimate</th><th scope="col">Lower bound</th><th scope="col">Upper bound</th></tr></thead>
<tbody id="perf-rows"></tbody>
</table>
</div>
</fieldset>

<fieldset>
<legend>Confusion matrix counts (optional)</legend>
<p class="tcv-hint">Shown in the draft only when all four counts are entered.</p>
<div class="tcv-grid tcv-grid--4">
<div class="tcv-field"><label for="f-tp">True positives</label><input id="f-tp" name="tp" type="number" min="0" step="1" inputmode="numeric"></div>
<div class="tcv-field"><label for="f-fp">False positives</label><input id="f-fp" name="fp" type="number" min="0" step="1" inputmode="numeric"></div>
<div class="tcv-field"><label for="f-fn">False negatives</label><input id="f-fn" name="fn" type="number" min="0" step="1" inputmode="numeric"></div>
<div class="tcv-field"><label for="f-tn">True negatives</label><input id="f-tn" name="tn" type="number" min="0" step="1" inputmode="numeric"></div>
</div>
<div class="tcv-field"><label for="f-threshold">Decision threshold and how it was chosen</label><input id="f-threshold" name="threshold" type="text" autocomplete="off"></div>
</fieldset>

<fieldset>
<legend>Subgroups, limitations and notes</legend>
<div class="tcv-field"><label for="f-subgroups">Subgroup results, one per line, as you will report them</label><textarea id="f-subgroups" name="subgroups" rows="3"></textarea></div>
<div class="tcv-field"><label for="f-limitations">Limitations</label><textarea id="f-limitations" name="limitations" rows="3"></textarea></div>
<div class="tcv-field"><label for="f-notes">Other notes</label><textarea id="f-notes" name="notes" rows="2"></textarea></div>
</fieldset>

<p id="tcv-report-status" class="tcv-status-msg" role="status" aria-live="polite"></p>

<div class="tcv-actions">
<button type="submit" class="md-button md-button--primary">Generate draft</button>
<button type="button" id="tcv-download-md" class="md-button" disabled>Download Markdown</button>
<button type="button" id="tcv-download-html" class="md-button" disabled>Download HTML</button>
</div>

</form>

<section id="tcv-report-output" class="tcv-report" aria-labelledby="tcv-report-output-title" hidden>
<h2 id="tcv-report-output-title" tabindex="-1">Draft preview</h2>
<div id="tcv-report-preview"></div>
</section>

<noscript><p>This tool needs JavaScript. Without it, write the summary by hand using the items listed on the Documenting validation page.</p></noscript>

<script src="../assets/js/regulatory-report.js"></script>
