/* Quiz engine and simulated scenarios for the Learn pages.
 *
 * - [data-tcv-quiz="full" | "tutorial" | "tutorial-final"]: renders a question set (quiz page,
 *   tutorials page).
 * - #tcv-scenarios: simulated validation scenarios (visualisations page).
 *   Fold scores are drawn with Math.random() around values chosen for
 *   illustration. They are not results from a model or a dataset.
 */
(function () {
  "use strict";

  var CATEGORIES = {
    leakage: "Data leakage",
    method: "Choosing a method",
    clinical: "Clinical application",
    imaging: "Medical imaging",
    temporal: "Temporal data"
  };

  var QUIZ = {
    full: [
      { scenario: "ICU mortality prediction", category: "leakage",
        question: "You are predicting ICU mortality from hourly measurements over 48 hours for 1,000 patients. What is the most critical validation concern?",
        options: ["Having enough data points", "Balancing positive and negative cases", "Preventing temporal leakage: using information recorded after the prediction time", "Using the right machine learning algorithm"],
        correct: 2,
        explanation: "If features include measurements taken after the prediction time (or after the outcome), the model is scored on information it would not have at the bedside, and the estimate is too high.",
        insight: "Define the prediction time first, build features only from data before it, and validate forward in time." },
      { scenario: "Multi-site study", category: "method",
        question: "You have data from 6 hospitals with different populations and measurement protocols. You want to know how the model will do at a new hospital. Which approach fits?",
        options: ["Standard 5-fold cross-validation", "Stratified 5-fold cross-validation", "Leave-one-site-out cross-validation", "Bootstrap validation"],
        correct: 2,
        explanation: "Holding out a whole hospital at a time tests transfer to a site the model has not seen. Row-level folds mix every hospital into training and test.",
        insight: "In trustcv: LeaveOneGroupOut with groups=site, or HierarchicalGroupKFold(hierarchy_level=\"hospital\")." },
      { scenario: "Longitudinal patient study", category: "clinical",
        question: "Your dataset has about 4 visits per patient over 2 years (500 patients). How should you split it?",
        options: ["Random split of all visits", "By patient ID: all visits of a patient in the same fold", "By visit date", "Stratify by number of visits per patient"],
        correct: 1,
        explanation: "Visits of one patient share stable characteristics. If they are on both sides, the model can recognise the patient instead of learning the condition.",
        insight: "Pass the patient IDs as groups= (for example StratifiedGroupKFold). trustcv's group-overlap check then confirms no patient is on both sides." },
      { scenario: "Diagnostic test development", category: "clinical",
        question: "A new test is validated on 200 positive and 1,800 negative cases and reaches 95% accuracy. What should you check?",
        options: ["The accuracy seems too low", "Accuracy is only 5 points above the 90% of always predicting negative; look at sensitivity and specificity", "The dataset is too small", "You need more features"],
        correct: 1,
        explanation: "With 90% negatives, a test that always says negative is 90% accurate. 95% accuracy means 100 errors out of 2,000: if they were all missed positives, sensitivity would be 50%; if they were all false alarms, it would be 100%. Accuracy alone cannot tell these apart.",
        insight: "Report sensitivity, specificity (or PPV and NPV) and a threshold-free metric such as ROC AUC, each with an interval." },
      { scenario: "Real-time sepsis prediction", category: "method",
        question: "A model uses the last 6 hours of data to predict sepsis in the next 2 hours. How should you validate it?",
        options: ["Expanding window validation without a gap", "Rolling-window validation with a 2-hour gap between training and test", "Random time-based splits", "Bootstrap sampling of time points"],
        correct: 1,
        explanation: "A gap the length of the prediction horizon stops labels that overlap the test period from leaking into training, and the window mirrors how the model will be used.",
        insight: "RollingWindowCV and ExpandingWindowCV take gap=, counted in rows. Keep patients whole as well when the data have several patients." },
      { scenario: "Genomics and clinical data", category: "leakage",
        question: "Adding one clinical feature raises AUC from 0.72 to 0.95. What should you investigate first?",
        options: ["Whether the feature has missing values", "Whether the feature contains information from after the outcome", "The feature's correlation with genetics", "Whether you need more data"],
        correct: 1,
        explanation: "A jump that large from one feature is a classic sign of target leakage: the feature may be recorded after, or because of, the diagnosis.",
        insight: "Check when each feature becomes available. trustcv's target-proxy check flags features that are suspiciously close to the label, but only a person can confirm when a value was recorded." },
      { scenario: "Small clinical dataset", category: "method",
        question: "With 150 samples, 5-fold CV gives AUCs from 0.6 to 0.9 across folds. What is a sensible next step?",
        options: ["Switch to leave-one-out", "Repeat stratified 5-fold with different shuffles and report the spread and an interval", "Reduce to 3-fold", "Use a single train-test split"],
        correct: 1,
        explanation: "With small test folds, fold-to-fold variation is expected. Repeating the split averages out the luck of one partition, and an interval shows how uncertain the estimate is. Leave-one-out has high variance, and a single split is the least stable of all.",
        insight: "The .632 bootstrap is another option with lower variance, but it can be optimistic for models that overfit. trustcv reports corrected_t intervals by default, which account for overlapping training sets." },
      { scenario: "Global health study", category: "clinical",
        question: "A malaria model is trained on data from Kenya. You want to know how it performs in Nigeria. What do you need?",
        options: ["10-fold cross-validation on Kenyan data", "Leave-one-out on Kenyan data", "A test on a separate Nigerian dataset", "Bootstrap validation on combined Kenya-Nigeria data"],
        correct: 2,
        explanation: "Cross-validation inside one country estimates performance in that country. Populations, parasites and health systems differ, so transfer has to be tested on data from the target setting.",
        insight: "Internal cross-validation and external validation answer different questions; report both." },
      { scenario: "Preprocessing", category: "leakage",
        question: "You standardise all features with a scaler fitted on the full dataset, then run cross-validation. What is the problem?",
        options: ["None: scaling does not use the labels", "The scaler saw the rows that later act as test data, so test-fold information reaches training", "Scaling should happen after the model is fitted", "You need more folds"],
        correct: 1,
        explanation: "Anything fitted on the full dataset before splitting has seen the test folds. Put the scaler inside a Pipeline so it is refitted on each training fold.",
        insight: "For plain scaling the bias is usually small; for imputation, feature selection and resampling it can be large. No check can see preprocessing done to X before validate(), so the Pipeline habit matters." },
      { scenario: "Clinical decision support", category: "clinical",
        question: "A model shows 92% accuracy in validation but 78% after deployment. What is the most likely explanation?",
        options: ["A different patient population in deployment", "Data leakage during validation", "Overfitting to the training data", "Any of these is possible"],
        correct: 3,
        explanation: "Population shift, leakage that inflated the validation estimate, and overfitting (especially through tuning on the test folds) can each produce a gap like this.",
        insight: "Validation can rule out some of these causes (leakage, tuning on test data); only data from the deployment setting can rule out shift." },
      { scenario: "Chest X-ray", category: "imaging",
        question: "You train a pneumonia detector on 10,000 X-rays from 2,500 patients (about 4 per patient). What is the key validation principle?",
        options: ["Split at image level to use the most data", "Split at patient level", "Stratify by pneumonia presence only", "Balance images per patient across folds"],
        correct: 1,
        explanation: "Images of one patient share anatomy, devices and disease course. Splitting by image lets the model recognise patients it has already seen.",
        insight: "Use groups=patient_ids with StratifiedGroupKFold to keep patients whole and balance the outcome." },
      { scenario: "Laboratory medicine", category: "temporal",
        question: "You predict abnormal lab results from a 5-year dataset. Test protocols changed in year 3. The model will be used under the new protocol. How should you validate?",
        options: ["Random 5-fold cross-validation across all years", "Forward split: train on the earlier years, test on the later years", "Report performance before and after the change separately, without a forward split", "Bootstrap sampling from each year equally"],
        correct: 1,
        explanation: "Deployment will see post-change data. Training on earlier years and testing on later years measures how the model copes with the shift. Reporting results per period is a useful addition, but on its own it does not test transfer across the change.",
        insight: "Practice, equipment and coding change over time. A forward split exposes that drift; random folds hide it." },
      { scenario: "Emergency department triage", category: "leakage",
        question: "A triage model uses symptoms, vitals and a 'chief_complaint_processed' field that is suspiciously predictive. What should you verify?",
        options: ["Whether the field has enough vocabulary coverage", "Whether the triage decision influenced how the field was created", "Whether the field correlates with demographics", "Whether stop words were removed"],
        correct: 1,
        explanation: "If staff recorded or coded the complaint after deciding the triage level, the field carries the outcome. That is target leakage.",
        insight: "Fields written by people who already knew the outcome are a common, hard-to-see source of leakage." },
      { scenario: "Wearable monitoring", category: "temporal",
        question: "A wearable records heart rate every minute; you have 6 months per user. For arrhythmia detection you should:",
        options: ["Split randomly within each user's timeline", "Keep users separate and validate forward in time", "Take every 5th day as validation across users", "Use each user's last month as test data, with the same users in training"],
        correct: 1,
        explanation: "The data have both user-level and time dependence. Holding out users tests new people; ordering by time avoids training on the future.",
        insight: "PurgedGroupTimeSeriesSplit with groups=user_ids and group_exclusive=True combines both." },
      { scenario: "Pharmacogenomics", category: "method",
        question: "A model with genetic variants reaches 0.95 AUC; a simple model with age and weight reaches 0.91. What does this tell you?",
        options: ["The genetic data clearly add a meaningful improvement", "The gain is small; check it against its uncertainty, with any feature selection inside nested CV, before crediting the genetic model", "Genetics should predict metabolism perfectly", "You should add more genetic variants"],
        correct: 1,
        explanation: "A 0.04 difference may be real, noise, or optimism from selecting variants on the same folds. These numbers alone show neither a real gain nor overfitting.",
        insight: "Compare models on the same folds with an interval for the difference, and keep variant selection inside each training fold." },
      { scenario: "Digital pathology", category: "imaging",
        question: "You have 10,000 tissue patches from 500 patients (20 per patient). What is the most critical validation consideration?",
        options: ["Equal numbers of cancer and normal patches", "Splitting at patient level", "Stratifying by tissue type", "Maximising patches in training"],
        correct: 1,
        explanation: "Patches from one patient share staining, scanner and disease state. Mixing them across training and test overstates performance on new patients.",
        insight: "Group by patient, and by slide or scanner too if you want to test transfer across labs." },
      { scenario: "Electronic health records", category: "temporal",
        question: "A readmission model uses EHR data from 2015-2023. The hospital changed EHR systems in 2019. What validation addresses this?",
        options: ["Random 80/20 split across all years", "Forward split: earlier years for training, later years for testing", "Stratified split keeping yearly proportions", "Leave-one-year-out cross-validation"],
        correct: 1,
        explanation: "The system change shifts how data are recorded. A forward split shows whether a model built on older data holds up on newer data.",
        insight: "Look for such breakpoints before choosing a split; they are often documented only in IT change logs." },
      { scenario: "Adverse events across trial sites", category: "clinical",
        question: "A model scores AUC 0.85 in cross-validation but 0.65 at some trial sites. What is the likely cause?",
        options: ["Too little training data overall", "Site-specific effects that the validation did not test", "The model is too simple", "Too few adverse events"],
        correct: 1,
        explanation: "If every fold contained every site, the validation never tested a new site. Leave-one-site-out would have shown the spread across sites.",
        insight: "Report per-site results as well as the average; the worst site often matters most." },
      { scenario: "MRI sequences", category: "imaging",
        question: "A tumour segmentation model uses T1, T2 and FLAIR. Performance drops sharply when any sequence is missing. This indicates:",
        options: ["The model is fine: all sequences are always needed", "The model depends on complete inputs, so validation should include realistic missing-sequence cases", "You only need more data with missing sequences", "The architecture must be replaced"],
        correct: 1,
        explanation: "If missing sequences happen in practice, a validation set without them overstates real-world performance. Test the conditions the model will meet.",
        insight: "Match the test data to deployment: protocols, missing inputs and image quality." },
      { scenario: "Metabolomics biomarkers", category: "method",
        question: "From 200 patients, feature selection keeps 50 metabolites and the model reaches 95% accuracy. What is concerning?",
        options: ["95% accuracy is always unrealistic", "Selecting features on the same data used for evaluation makes the estimate optimistic", "You must reduce to fewer than 10 metabolites", "You should validate on healthy controls only"],
        correct: 1,
        explanation: "When features are chosen using all the data, the test folds influenced the choice. Feature selection belongs inside each training fold, with nested CV for any tuning, to get a less biased estimate.",
        insight: "Put the selector in the Pipeline; trustcv cannot detect selection done before validate()." }
    ],
    tutorial: [
      { scenario: "Quick check", category: "leakage",
        question: "Why does standard K-fold cause leakage with repeated patient visits?",
        options: ["It does not shuffle the data", "It puts visits of the same patient in both training and test", "It uses too few folds"],
        correct: 1,
        explanation: "The model can memorise patient-specific patterns from training visits and recognise them in test visits of the same patient." }
    ],
    "tutorial-final": [
      { scenario: "Question 1", category: "temporal",
        question: "You have ICU data with hourly measurements and want to predict future deterioration. Which splitter?",
        options: ["Standard K-fold", "A forward-only time split such as TimeSeriesSplit", "Leave-one-out"],
        correct: 1,
        explanation: "Forward-only splits never train on rows recorded after the test rows. With several patients, also keep patients whole." },
      { scenario: "Question 2", category: "clinical",
        question: "Data from a multi-centre trial, and you want to know how the model transfers to a new centre. Use:",
        options: ["Random splits", "Leave-one-centre-out (LeaveOneGroupOut with groups=centre)", "Bootstrap validation"],
        correct: 1,
        explanation: "Holding out a centre at a time tests transfer to a site not seen in training." },
      { scenario: "Question 3", category: "leakage",
        question: "How much can leakage inflate a performance estimate?",
        options: ["Never more than a few percent", "It can be large, and the size depends on the data and the leak", "Always exactly 20-40%"],
        correct: 1,
        explanation: "There is no fixed size. A leak can add almost nothing or turn a useless model into an apparently excellent one; the example above shows a large gap." }
    ]
  };

  /* ---------- Quiz ---------- */
  function renderQuiz(root) {
    var set = QUIZ[root.getAttribute("data-tcv-quiz")];
    if (!set) return;
    var prefix = "tcvq-" + root.getAttribute("data-tcv-quiz");
    var answered = {};
    root.innerHTML = "";

    var status = document.createElement("p");
    status.className = "tcv-quiz-status";
    status.setAttribute("role", "status");

    set.forEach(function (q, qi) {
      var fs = document.createElement("fieldset");
      fs.className = "tcv-quiz-q";
      var lg = document.createElement("legend");
      lg.innerHTML = "<span class=\"tcv-quiz-n\"></span> ";
      lg.querySelector(".tcv-quiz-n").textContent = (set.length > 4 ? (qi + 1) + ". " : "") + q.scenario;
      fs.appendChild(lg);
      var p = document.createElement("p");
      p.className = "tcv-quiz-text";
      p.textContent = q.question;
      fs.appendChild(p);
      q.options.forEach(function (opt, oi) {
        var id = prefix + "-" + qi + "-" + oi;
        var wrap = document.createElement("div");
        wrap.className = "tcv-quiz-opt";
        var input = document.createElement("input");
        input.type = "radio"; input.name = prefix + "-" + qi; input.id = id; input.value = String(oi);
        var label = document.createElement("label");
        label.setAttribute("for", id);
        label.textContent = opt;
        wrap.appendChild(input); wrap.appendChild(label);
        fs.appendChild(wrap);
      });
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "md-button tcv-quiz-check";
      btn.textContent = "Check answer";
      btn.setAttribute("aria-label", "Check answer to question " + (qi + 1));
      fs.appendChild(btn);
      var fb = document.createElement("div");
      fb.className = "tcv-quiz-feedback";
      fb.setAttribute("aria-live", "polite");
      fs.appendChild(fb);

      btn.addEventListener("click", function () {
        var sel = fs.querySelector("input:checked");
        fb.innerHTML = "";
        var head = document.createElement("p");
        head.className = "tcv-quiz-verdict";
        if (!sel) {
          head.textContent = "Choose an answer first.";
          fb.appendChild(head);
          return;
        }
        var ok = parseInt(sel.value, 10) === q.correct;
        answered[qi] = ok;
        fs.classList.toggle("is-correct", ok);
        fs.classList.toggle("is-incorrect", !ok);
        head.textContent = ok ? "Correct." : "Not quite. The best answer is: " + q.options[q.correct] + ".";
        fb.appendChild(head);
        var ex = document.createElement("p");
        ex.textContent = q.explanation;
        fb.appendChild(ex);
        if (q.insight) {
          var ins = document.createElement("p");
          ins.className = "tcv-quiz-insight";
          ins.textContent = q.insight;
          fb.appendChild(ins);
        }
        update();
      });
      root.appendChild(fs);
    });

    var summary = document.createElement("div");
    summary.className = "tcv-quiz-summary";
    root.appendChild(status);
    root.appendChild(summary);

    var reset = document.createElement("button");
    reset.type = "button";
    reset.className = "md-button";
    reset.textContent = "Start again";
    reset.addEventListener("click", function () { renderQuiz(root); root.scrollIntoView(); });
    root.appendChild(reset);

    function update() {
      var keys = Object.keys(answered), right = keys.filter(function (k) { return answered[k]; }).length;
      status.textContent = right + " correct out of " + keys.length + " checked (" + set.length + " questions).";
      summary.innerHTML = "";
      if (keys.length < set.length || set.length < 5) return;
      var h = document.createElement("h3");
      h.textContent = "Your results by topic";
      summary.appendChild(h);
      var table = document.createElement("table");
      table.innerHTML = "<thead><tr><th scope=\"col\">Topic</th><th scope=\"col\">Correct</th></tr></thead>";
      var tb = document.createElement("tbody");
      Object.keys(CATEGORIES).forEach(function (c) {
        var total = 0, ok = 0;
        set.forEach(function (q, i) { if (q.category === c) { total++; if (answered[i]) ok++; } });
        var tr = document.createElement("tr");
        var a = document.createElement("td"); a.textContent = CATEGORIES[c];
        var b = document.createElement("td"); b.textContent = ok + " of " + total;
        tr.appendChild(a); tr.appendChild(b); tb.appendChild(tr);
      });
      table.appendChild(tb);
      summary.appendChild(table);
    }
    update();
  }

  /* ---------- Simulated scenarios ---------- */
  var REPO = "https://github.com/ki-smile/trustcv/blob/main/";
  var SCENARIOS = [
    { key: "icu", title: "ICU mortality: future information", metric: "AUC",
      setting: "Imagined setting: hourly vital signs and labs from ICU patients; predict death within 24 hours.",
      code: "trustcv/examples/icu_patient_monitoring.py",
      methods: [
        { label: "Random row split (future rows in training)", mean: 0.90, spread: 0.04, sound: false, how: "KFoldMedical(shuffle=True) on time-ordered rows",
          text: "Rows from later hours of the same stay end up in training, so the model is scored with hindsight." },
        { label: "Forward-only split, patients kept whole", mean: 0.69, spread: 0.05, sound: true, how: "PurgedGroupTimeSeriesSplit(group_exclusive=True) with groups= and timestamps=",
          text: "Only earlier data are used to predict later outcomes, as at the bedside." }
      ] },
    { key: "sites", title: "Diabetes across 8 hospitals", metric: "AUC",
      setting: "Imagined setting: one model, eight hospitals with different populations and protocols.",
      code: "trustcv/examples/multisite_clinical_trial.py",
      methods: [
        { label: "K-fold over all rows", mean: 0.82, spread: 0.03, sound: false, how: "KFoldMedical",
          text: "Every hospital is in every training set, so the estimate says nothing about a new hospital." },
        { label: "Stratified K-fold", mean: 0.80, spread: 0.03, sound: false, how: "StratifiedKFoldMedical",
          text: "Balances the outcome but still mixes hospitals across training and test." },
        { label: "Group K-fold by hospital", mean: 0.74, spread: 0.07, sound: true, how: "GroupKFoldMedical with groups=hospital",
          text: "Each test fold contains hospitals the model has not seen." },
        { label: "Leave one hospital out", mean: 0.71, spread: 0.10, sound: true, how: "LeaveOneGroupOut with groups=hospital",
          text: "One fold per hospital; the spread across hospitals is part of the answer." }
      ] },
    { key: "sepsis", title: "Sepsis early warning over time", metric: "AUC",
      setting: "Imagined setting: 5-minute vital signs over 72 hours; warn 4 hours ahead.",
      code: "trustcv/examples/icu_patient_monitoring.py",
      methods: [
        { label: "Shuffled K-fold", mean: 0.88, spread: 0.03, sound: false, how: "KFoldMedical(shuffle=True)",
          text: "Neighbouring time points sit on both sides of the split; the model interpolates instead of forecasting." },
        { label: "Purged K-fold", mean: 0.80, spread: 0.04, sound: false, how: "PurgedKFoldCV(purge_gap=...)",
          text: "Purging removes rows next to the test block, but training still includes later blocks, so this is not a forward-only estimate." },
        { label: "Expanding window", mean: 0.74, spread: 0.05, sound: true, how: "ExpandingWindowCV(gap=...)",
          text: "Train on all earlier data, predict the next window. gap= is counted in rows." },
        { label: "Rolling window", mean: 0.73, spread: 0.05, sound: true, how: "RollingWindowCV(window_size=..., gap=...)",
          text: "A fixed-length recent window; useful when old data no longer reflect current practice." }
      ] },
    { key: "visits", title: "Repeated visits per patient", metric: "AUC",
      setting: "Imagined setting: 800 cardiology patients with 3-8 visits each; predict a cardiovascular event.",
      code: "trustcv/examples/data_leakage_detection_demo.py",
      methods: [
        { label: "Visit-level split", mean: 0.86, spread: 0.03, sound: false, how: "KFoldMedical",
          text: "Visits of one patient appear in training and test, so the model can recognise patients." },
        { label: "Patient-level split", mean: 0.72, spread: 0.05, sound: true, how: "GroupKFoldMedical with groups=patient_ids",
          text: "All visits of a patient stay together, as when the model meets new patients." },
        { label: "Stratified patient-level split", mean: 0.73, spread: 0.04, sound: true, how: "StratifiedGroupKFold with groups=patient_ids",
          text: "Patients kept whole and the outcome balanced across folds." }
      ] },
    { key: "imaging", title: "Skin lesion images", metric: "AUC",
      setting: "Imagined setting: several images per patient, melanoma to benign about 1 to 4.",
      code: "examples/pytorch_medical_cv.py",
      methods: [
        { label: "Image-level split", mean: 0.92, spread: 0.02, sound: false, how: "KFoldMedical over images",
          text: "Images of one patient on both sides; the model learns patient and device traits. A patient-level estimate in this simulation is about 0.73." },
        { label: "Patient-level split", mean: 0.74, spread: 0.04, sound: true, how: "GroupKFoldMedical with groups=patient_ids",
          text: "Tests generalisation to new patients." },
        { label: "Stratified patient-level split", mean: 0.76, spread: 0.04, sound: true, how: "StratifiedGroupKFold with groups=patient_ids",
          text: "Also keeps the melanoma share similar across folds." }
      ] },
    { key: "omics", title: "Multi-omics across batches", metric: "AUC",
      setting: "Imagined setting: tumour subtypes from RNA-seq and methylation, five sequencing batches, mixed ancestry.",
      code: null,
      methods: [
        { label: "Standard K-fold", mean: 0.88, spread: 0.05, sound: false, how: "KFoldMedical",
          text: "Batch and ancestry signals are shared by training and test, and can stand in for biology." },
        { label: "Ancestry-stratified CV (concept)", mean: 0.82, spread: 0.04, sound: false, how: "Not a trustcv method: StratifiedKFoldMedical on an ancestry-by-outcome label",
          text: "Balances ancestry across folds but does not test transfer to a new population." },
        { label: "Batch-aware CV (concept)", mean: 0.76, spread: 0.04, sound: true, how: "Not a trustcv method: implement with StratifiedGroupKFold(groups=batch_ids)",
          text: "Whole batches are held out, so batch artefacts cannot help." },
        { label: "Leave one population out", mean: 0.71, spread: 0.08, sound: true, how: "LeaveOneGroupOut with groups=population",
          text: "Tests transfer to an ancestry group not seen in training." }
      ] },
    { key: "drug", title: "Molecular property prediction", metric: "R²",
      setting: "Imagined setting: bioactivity of tens of thousands of compounds against 20 targets.",
      code: null,
      methods: [
        { label: "Random split", mean: 0.76, spread: 0.03, sound: false, how: "KFoldMedical",
          text: "Close analogues of test molecules are in training, which flatters the model." },
        { label: "Scaffold split (concept)", mean: 0.63, spread: 0.04, sound: true, how: "Not a trustcv method: implement with StratifiedGroupKFold(groups=scaffold_ids), or GroupKFoldMedical for a continuous target",
          text: "Test molecules have scaffolds the model has not seen, closer to finding new chemical matter." },
        { label: "Time split", mean: 0.68, spread: 0.04, sound: true, how: "TimeSeriesSplit with timestamps= (compound registration date)",
          text: "Predicts later compounds from earlier ones, as in a running project." },
        { label: "Target-based CV (concept)", mean: 0.71, spread: 0.05, sound: true, how: "Not a trustcv method: implement with StratifiedGroupKFold(groups=target_ids)",
          text: "Holds out whole protein targets to test transfer across targets." }
      ] },
    { key: "epi", title: "Outbreak prediction across regions", metric: "Sensitivity",
      setting: "Imagined setting: weekly case counts for 200 cities in 10 countries; predict outbreaks two weeks ahead.",
      code: "examples/disease_spread_modeling.py",
      methods: [
        { label: "Time series split only", mean: 0.82, spread: 0.04, sound: false, how: "TimeSeriesSplit",
          text: "Respects time but trains on neighbouring cities of the test cities, which share outbreaks." },
        { label: "Spatial block CV", mean: 0.74, spread: 0.05, sound: true, how: "SpatialBlockCV, coordinates passed to split()",
          text: "Holds out whole regions; nearby cities are not on both sides." },
        { label: "Leave one country out", mean: 0.68, spread: 0.08, sound: true, how: "LeaveOneGroupOut with groups=country",
          text: "Tests transfer to a country not seen in training." },
        { label: "Spatiotemporal block CV", mean: 0.70, spread: 0.05, sound: true, how: "SpatiotemporalBlockCV with coordinates= and timestamps=",
          text: "Holds out a region in a time period, with optional buffers in space and time." }
      ] }
  ];

  function renderScenarios(root) {
    var sSel = document.getElementById("tcv-scn-scenario");
    var mSel = document.getElementById("tcv-scn-method");
    var run = document.getElementById("tcv-scn-run");
    var out = document.getElementById("tcv-scn-out");
    if (!sSel || !mSel || !run || !out) return;

    SCENARIOS.forEach(function (s) {
      var o = document.createElement("option"); o.value = s.key; o.textContent = s.title; sSel.appendChild(o);
    });
    function scenario() { return SCENARIOS.filter(function (s) { return s.key === sSel.value; })[0] || SCENARIOS[0]; }
    function fillMethods() {
      mSel.innerHTML = "";
      scenario().methods.forEach(function (m, i) {
        var o = document.createElement("option"); o.value = String(i); o.textContent = m.label; mSel.appendChild(o);
      });
    }
    function describe() {
      var s = scenario();
      var setting = root.querySelector(".tcv-scn-setting");
      if (setting) {
        setting.textContent = s.setting + " ";
        if (s.code) {
          var a = document.createElement("a");
          a.href = REPO + s.code;
          a.textContent = "Related code: " + s.code;
          setting.appendChild(a);
        }
      }
    }
    function simulate() {
      var s = scenario(), m = s.methods[parseInt(mSel.value, 10) || 0];
      var folds = [];
      for (var i = 0; i < 5; i++) folds.push(Math.max(0, Math.min(1, m.mean + (Math.random() * 2 - 1) * m.spread)));
      var mean = folds.reduce(function (a, b) { return a + b; }, 0) / folds.length;
      out.innerHTML = "";

      var head = document.createElement("p");
      head.innerHTML = "<span class=\"tcv-sim-badge\">Simulated</span> ";
      var strong = document.createElement("strong");
      strong.textContent = m.label + ": mean " + s.metric + " " + mean.toFixed(2);
      head.appendChild(strong);
      head.appendChild(document.createTextNode(m.sound ? " (matches the question being asked)" : " (likely optimistic for this question)"));
      out.appendChild(head);

      var bars = document.createElement("div");
      bars.className = "tcv-bars";
      bars.setAttribute("role", "img");
      bars.setAttribute("aria-label", "Simulated fold scores: " + folds.map(function (f, i) { return "fold " + (i + 1) + " " + f.toFixed(2); }).join(", "));
      folds.forEach(function (f, i) {
        var r = document.createElement("div");
        r.className = "tcv-bar-row";
        r.setAttribute("aria-hidden", "true");
        var l = document.createElement("span"); l.className = "tcv-bar-label"; l.textContent = "Fold " + (i + 1);
        var track = document.createElement("span"); track.className = "tcv-bar-track";
        var bar = document.createElement("span"); bar.className = "tcv-bar" + (m.sound ? "" : " tcv-bar-optimistic"); bar.style.width = (f * 100).toFixed(1) + "%";
        track.appendChild(bar);
        var v = document.createElement("span"); v.className = "tcv-bar-value"; v.textContent = f.toFixed(2);
        r.appendChild(l); r.appendChild(track); r.appendChild(v);
        bars.appendChild(r);
      });
      out.appendChild(bars);

      var p = document.createElement("p"); p.textContent = m.text; out.appendChild(p);
      var how = document.createElement("p");
      how.innerHTML = "<strong>In trustcv:</strong> ";
      var code = document.createElement("code"); code.textContent = m.how; how.appendChild(code);
      out.appendChild(how);
      var note = document.createElement("p");
      note.className = "tcv-sim-note";
      note.textContent = "Illustrative only: the five fold scores are random draws around values chosen to show the typical direction of the effect. They are not results from a model or a dataset, and the size of a real gap depends on your data.";
      out.appendChild(note);
    }

    sSel.addEventListener("change", function () { fillMethods(); describe(); out.innerHTML = ""; });
    mSel.addEventListener("change", function () { out.innerHTML = ""; });
    run.addEventListener("click", simulate);
    fillMethods();
    describe();
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-tcv-quiz]"), renderQuiz);
    var scn = document.getElementById("tcv-scenarios");
    if (scn) renderScenarios(scn);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
