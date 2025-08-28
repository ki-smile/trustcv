/**
 * Enhanced Interactive Tutorials and Quiz System
 */

// Demo management
let currentDemo = 'data-leakage';

// Quiz state
let quizData = [];
let currentQuizQuestion = 0;
let userAnswers = [];
let quizStarted = false;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeTutorials();
    initializeQuiz();
});

function initializeTutorials() {
    // Set up smooth scrolling
    setupSmoothScrolling();
    
    // Only initialize demos if we're on demonstrations page
    if (document.querySelector('.demonstrations')) {
        // Initialize demo tabs
        showDemo('data-leakage');
        
        // Set up visualization containers
        setupVisualizationContainers();
    }
}

function setupSmoothScrolling() {
    window.scrollToSection = function(sectionId) {
        const element = document.getElementById(sectionId);
        if (element) {
            element.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    };
}

function setupVisualizationContainers() {
    // Ensure Plotly containers are properly sized
    const containers = ['leakage-plot', 'cv-comparison-plot', 'temporal-demo-plot', 'patient-demo-plot', 
                       'imaging-demo-plot', 'genomics-demo-plot', 'drug-discovery-plot', 'epidemiology-plot'];
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            container.style.width = '100%';
            container.style.height = '400px';
        }
    });
}

// Demo Tab Management
function showDemo(demoId) {
    // Update active tab
    document.querySelectorAll('.tab-header').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Find and activate the clicked tab
    const clickedTab = document.querySelector(`[onclick="showDemo('${demoId}')"]`);
    if (clickedTab) {
        clickedTab.classList.add('active');
    }
    
    // Show corresponding demo content
    document.querySelectorAll('.demo-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const targetDemo = document.getElementById(demoId);
    if (targetDemo) {
        targetDemo.classList.add('active');
    }
    
    currentDemo = demoId;
    
    // Run initial demo if needed
    switch(demoId) {
        case 'data-leakage':
            runLeakageDemo();
            break;
        case 'cv-comparison':
            runCVComparison();
            break;
        case 'temporal-demo':
            runTemporalDemo();
            break;
        case 'patient-demo':
            runPatientDemo();
            break;
        case 'imaging-demo':
            runImagingDemo();
            break;
        case 'genomics-demo':
            runGenomicsDemo();
            break;
        case 'drug-discovery':
            runDrugDemo();
            break;
        case 'epidemiology-demo':
            runEpiDemo();
            break;
    }
}

// Data Leakage Demonstration
function runLeakageDemo() {
    const method = document.getElementById('leakage-method').value;
    
    // Simulate different performance based on method
    let auc, sensitivity, specificity, clinicalImpact;
    
    if (method === 'wrong') {
        // Inflated performance due to leakage
        auc = (0.85 + Math.random() * 0.1).toFixed(3);
        sensitivity = (0.82 + Math.random() * 0.08).toFixed(3);
        specificity = (0.87 + Math.random() * 0.08).toFixed(3);
        
        clinicalImpact = `
            <div class="impact-warning">
                ⚠️ <strong>DANGEROUSLY MISLEADING RESULTS</strong>
                <ul>
                    <li>Performance overestimated by ~25-40%</li>
                    <li>False confidence in model reliability</li>
                    <li>Could lead to inappropriate clinical deployment</li>
                    <li>Potential patient safety risks</li>
                </ul>
                <p><strong>Why?</strong> Future information leaked into training, making predictions artificially easy.</p>
            </div>
        `;
    } else {
        // Realistic performance with proper validation
        auc = (0.65 + Math.random() * 0.08).toFixed(3);
        sensitivity = (0.62 + Math.random() * 0.1).toFixed(3);
        specificity = (0.68 + Math.random() * 0.08).toFixed(3);
        
        clinicalImpact = `
            <div class="impact-success">
                ✅ <strong>REALISTIC, TRUSTWORTHY RESULTS</strong>
                <ul>
                    <li>True model performance accurately estimated</li>
                    <li>Appropriate confidence intervals</li>
                    <li>Safe for clinical decision making</li>
                    <li>Regulatory compliance maintained</li>
                </ul>
                <p><strong>Why?</strong> Temporal validation prevents look-ahead bias and future information leakage.</p>
            </div>
        `;
    }
    
    // Update metrics display
    document.getElementById('auc-value').textContent = auc;
    document.getElementById('sensitivity-value').textContent = sensitivity;
    document.getElementById('specificity-value').textContent = specificity;
    document.getElementById('clinical-impact-text').innerHTML = clinicalImpact;
    
    // Create visualization
    createLeakageVisualization(method, auc);
}

function createLeakageVisualization(method, aucValue) {
    const trace1 = {
        x: ['Wrong Method', 'Correct Method'],
        y: [method === 'wrong' ? aucValue : 0.65, method === 'wrong' ? 0.65 : aucValue],
        type: 'bar',
        name: 'AUC Score',
        marker: {
            color: method === 'wrong' ? ['#ff4444', '#cccccc'] : ['#cccccc', '#44aa44']
        }
    };
    
    const layout = {
        title: 'Performance Comparison: Impact of Data Leakage',
        yaxis: { title: 'AUC-ROC Score', range: [0.5, 1.0] },
        annotations: [{
            x: method === 'wrong' ? 0 : 1,
            y: parseFloat(aucValue) + 0.05,
            text: method === 'wrong' ? '⚠️ Inflated!' : '✅ Realistic',
            showarrow: false,
            font: { size: 14, color: method === 'wrong' ? '#ff4444' : '#44aa44' }
        }]
    };
    
    Plotly.newPlot('leakage-plot', [trace1], layout, {responsive: true});
}

// CV Comparison Demo
function runCVComparison() {
    const method = document.getElementById('cv-method-select').value;
    
    // Simulate different CV methods performance
    const hospitalData = generateHospitalData();
    const results = simulateCVMethod(method, hospitalData);
    
    // Create visualization
    createCVComparisonVisualization(results, method);
    
    // Update explanation
    updateCVExplanation(method, results);
}

function generateHospitalData() {
    const hospitals = ['Hospital A', 'Hospital B', 'Hospital C', 'Hospital D', 
                      'Hospital E', 'Hospital F', 'Hospital G', 'Hospital H'];
    
    return hospitals.map(hospital => ({
        name: hospital,
        patients: Math.floor(Math.random() * 800) + 200,
        prevalence: Math.random() * 0.3 + 0.1,
        performance: Math.random() * 0.2 + 0.7
    }));
}

function simulateCVMethod(method, hospitalData) {
    let meanScore, stdScore, explanation;
    
    switch(method) {
        case 'standard':
            meanScore = 0.82;
            stdScore = 0.04;
            explanation = "Standard K-Fold ignores hospital clustering, leading to optimistic estimates.";
            break;
        case 'stratified':
            meanScore = 0.79;
            stdScore = 0.05;
            explanation = "Stratified K-Fold maintains class balance but still ignores hospital effects.";
            break;
        case 'grouped':
            meanScore = 0.74;
            stdScore = 0.08;
            explanation = "Hospital-grouped K-Fold properly accounts for site-specific biases.";
            break;
        case 'leave-site-out':
            meanScore = 0.71;
            stdScore = 0.12;
            explanation = "Leave-One-Site-Out provides the most conservative and realistic estimates.";
            break;
    }
    
    return { meanScore, stdScore, explanation, method };
}

function createCVComparisonVisualization(results, method) {
    const methods = ['Standard K-Fold', 'Stratified K-Fold', 'Hospital-Grouped', 'Leave-Site-Out'];
    const scores = [0.82, 0.79, 0.74, 0.71];
    const errors = [0.04, 0.05, 0.08, 0.12];
    
    const currentIndex = ['standard', 'stratified', 'grouped', 'leave-site-out'].indexOf(method);
    
    const colors = methods.map((_, i) => i === currentIndex ? '#2196F3' : '#cccccc');
    
    const trace = {
        x: methods,
        y: scores,
        error_y: {
            type: 'data',
            array: errors,
            visible: true
        },
        type: 'bar',
        marker: { color: colors }
    };
    
    const layout = {
        title: 'Cross-Validation Method Comparison',
        yaxis: { title: 'AUC Score', range: [0.6, 0.9] },
        xaxis: { tickangle: -45 }
    };
    
    Plotly.newPlot('cv-comparison-plot', [trace], layout, {responsive: true});
}

function updateCVExplanation(method, results) {
    const explanationDiv = document.getElementById('cv-comparison-explanation');
    
    explanationDiv.innerHTML = `
        <h4>Current Method: ${method.charAt(0).toUpperCase() + method.slice(1)} CV</h4>
        <p><strong>Performance:</strong> ${results.meanScore.toFixed(3)} ± ${results.stdScore.toFixed(3)}</p>
        <p><strong>Interpretation:</strong> ${results.explanation}</p>
        <div class="method-recommendation">
            <h5>💡 Recommendation:</h5>
            <p>${getMethodRecommendation(method)}</p>
        </div>
    `;
}

function getMethodRecommendation(method) {
    const recommendations = {
        'standard': 'Avoid for multi-site studies. Use grouped methods instead.',
        'stratified': 'Better than standard but still doesn\'t handle site effects.',
        'grouped': 'Good choice for multi-site validation with reasonable computational cost.',
        'leave-site-out': 'Most rigorous for multi-site studies but requires sufficient sites.'
    };
    return recommendations[method];
}

// Temporal Demo
function runTemporalDemo() {
    const method = document.getElementById('temporal-method').value;
    const horizon = document.getElementById('prediction-horizon').value;
    
    // Update horizon display
    document.getElementById('horizon-value').textContent = horizon;
    
    // Generate temporal data visualization
    createTemporalVisualization(method, horizon);
    
    // Update explanation
    updateTemporalExplanation(method, horizon);
}

function createTemporalVisualization(method, horizon) {
    const timePoints = Array.from({length: 72}, (_, i) => i); // 72 hours
    let trainData, testData;
    
    switch(method) {
        case 'expanding':
            trainData = timePoints.slice(0, 48);
            testData = timePoints.slice(48, 48 + parseInt(horizon));
            break;
        case 'rolling':
            trainData = timePoints.slice(24, 48);
            testData = timePoints.slice(48, 48 + parseInt(horizon));
            break;
        case 'blocked':
            trainData = timePoints.filter(t => t % 12 < 8); // 8 hours train, 4 hours gap
            testData = timePoints.filter(t => t % 12 >= 8);
            break;
        case 'purged':
            trainData = timePoints.slice(0, 40);
            testData = timePoints.slice(48, 48 + parseInt(horizon)); // 8-hour gap
            break;
    }
    
    const trace1 = {
        x: trainData,
        y: trainData.map(() => 1),
        name: 'Training Data',
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'blue', size: 4 }
    };
    
    const trace2 = {
        x: testData,
        y: testData.map(() => 0.5),
        name: 'Test Data',
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'red', size: 4 }
    };
    
    const layout = {
        title: `Temporal Validation: ${method.charAt(0).toUpperCase() + method.slice(1)} Method`,
        xaxis: { title: 'Time (hours)' },
        yaxis: { title: 'Data Split', showticklabels: false },
        showlegend: true
    };
    
    Plotly.newPlot('temporal-demo-plot', [trace1, trace2], layout, {responsive: true});
}

function updateTemporalExplanation(method, horizon) {
    const explanations = {
        'expanding': `Expanding window uses all historical data for training. Good for trends but may include outdated patterns.`,
        'rolling': `Rolling window uses recent ${24} hours only. Better for changing patterns but less historical context.`,
        'blocked': `Blocked time series creates separate training blocks with gaps. Prevents temporal leakage effectively.`,
        'purged': `Purged K-Fold removes ${8} hours around test periods to prevent information leakage.`
    };
    
    document.getElementById('temporal-explanation').innerHTML = `
        <h4>Method: ${method.charAt(0).toUpperCase() + method.slice(1)} Window</h4>
        <p><strong>Prediction Horizon:</strong> ${horizon} hours ahead</p>
        <p><strong>How it works:</strong> ${explanations[method]}</p>
        <div class="clinical-context">
            <h5>🏥 Clinical Application:</h5>
            <p>In ICU monitoring, this method ensures your sepsis prediction model only uses information available at prediction time, preventing look-ahead bias that could lead to overoptimistic performance estimates.</p>
        </div>
    `;
}

// Patient Demo
function runPatientDemo() {
    const method = document.getElementById('grouping-method').value;
    
    // Create patient grouping visualization
    createPatientVisualization(method);
    
    // Update explanation
    updatePatientExplanation(method);
}

function createPatientVisualization(method) {
    // Simulate patient visit data
    const patients = Array.from({length: 20}, (_, i) => `P${i+1}`);
    const visits = [];
    
    patients.forEach(patient => {
        const numVisits = Math.floor(Math.random() * 6) + 3; // 3-8 visits
        for (let v = 0; v < numVisits; v++) {
            visits.push({
                patient: patient,
                visit: v + 1,
                fold: method === 'visit-level' ? Math.floor(Math.random() * 5) : 
                      Math.floor(patients.indexOf(patient) / 4) // Patient-level grouping
            });
        }
    });
    
    const folds = [0, 1, 2, 3, 4];
    const traces = folds.map(fold => ({
        x: visits.filter(v => v.fold === fold).map(v => v.patient),
        y: visits.filter(v => v.fold === fold).map(v => v.visit),
        name: `Fold ${fold + 1}`,
        type: 'scatter',
        mode: 'markers',
        marker: { size: 8 }
    }));
    
    const layout = {
        title: `Patient Grouping Strategy: ${method.replace('-', ' ').toUpperCase()}`,
        xaxis: { title: 'Patient ID' },
        yaxis: { title: 'Visit Number' },
        showlegend: true
    };
    
    Plotly.newPlot('patient-demo-plot', traces, layout, {responsive: true});
}

function updatePatientExplanation(method) {
    const explanations = {
        'visit-level': `❌ INCORRECT: Visits from same patient scattered across folds. This causes data leakage as patient-specific patterns are shared between training and test sets.`,
        'patient-level': `✅ CORRECT: All visits from each patient stay in the same fold. This prevents patient-level information leakage and provides realistic performance estimates.`,
        'stratified-patient': `✅ ADVANCED: Patient-level grouping while maintaining outcome balance across folds. Best for imbalanced datasets.`
    };
    
    const leakageRisk = method === 'visit-level' ? 'HIGH RISK' : 'LOW RISK';
    const leakageColor = method === 'visit-level' ? 'style="color: #ff4444;"' : 'style="color: #44aa44;"';
    
    document.getElementById('patient-explanation').innerHTML = `
        <h4>Grouping Method: ${method.replace('-', ' ').charAt(0).toUpperCase() + method.replace('-', ' ').slice(1)}</h4>
        <p><strong>Data Leakage Risk:</strong> <span ${leakageColor}>${leakageRisk}</span></p>
        <p><strong>Explanation:</strong> ${explanations[method]}</p>
        <div class="regulatory-note">
            <h5>📋 Regulatory Perspective:</h5>
            <p>${method === 'visit-level' ? 
                'FDA/EMA would likely reject models validated this way due to inflated performance estimates.' :
                'This approach meets regulatory standards for clinical validation and deployment.'}</p>
        </div>
    `;
}

// Quiz System
function initializeQuiz() {
    // Only initialize if we're on the quiz page
    if (!document.getElementById('interactive-quiz')) {
        return;
    }
    
    quizData = [
        {
            scenario: "🏥 ICU Mortality Prediction",
            question: "You're developing a model to predict ICU mortality using patient data collected over 48 hours. Your dataset contains hourly measurements for 1,000 patients. What's the most critical validation concern?",
            options: [
                "Having enough data points",
                "Balancing positive and negative cases",
                "Preventing temporal data leakage", 
                "Using the right machine learning algorithm"
            ],
            correct: 2,
            explanation: "Temporal data leakage occurs when future information is used to predict past events. In ICU data, using measurements taken after the outcome (mortality) to predict that outcome creates unrealistic performance estimates.",
            insight: "In clinical practice, you only have access to current and historical data when making predictions. Proper temporal validation ensures your model performance reflects real-world deployment scenarios.",
            category: "leakage"
        },
        {
            scenario: "🏥 Multi-Site Clinical Trial",
            question: "You have data from 6 hospitals for a diabetes prediction model. Each hospital has different patient populations and measurement protocols. What's the best cross-validation approach?",
            options: [
                "Standard 5-fold cross-validation",
                "Stratified 5-fold cross-validation", 
                "Leave-one-site-out cross-validation",
                "Bootstrap validation"
            ],
            correct: 2,
            explanation: "Leave-one-site-out validation tests model generalizability across different hospital settings, which is crucial for multi-site deployment.",
            insight: "Hospital-specific biases (different populations, protocols, equipment) can dramatically affect model performance. Testing generalization across sites is essential for real-world deployment.",
            category: "method"
        },
        {
            scenario: "👥 Longitudinal Patient Study", 
            question: "Your dataset contains multiple visits per patient over 2 years (avg 4 visits per patient, 500 patients total). How should you split the data?",
            options: [
                "Random split of all visits",
                "Split by patient ID (all visits from same patient in same fold)",
                "Split by visit date",
                "Stratify by number of visits per patient"
            ],
            correct: 1,
            explanation: "Patient-level splitting prevents data leakage by ensuring all records from the same patient stay together. Random visit splitting would leak patient-specific patterns between train and test sets.",
            insight: "Patients have unique characteristics (genetics, lifestyle, comorbidities) that persist across visits. Mixing visits from the same patient between training and testing creates unrealistic performance estimates.",
            category: "clinical"
        },
        {
            scenario: "🔬 Diagnostic Test Development",
            question: "You're validating a new diagnostic test with 200 positive and 1,800 negative cases. The test achieves 95% accuracy with standard cross-validation. What's concerning?",
            options: [
                "The accuracy seems too low",
                "Need to check if the model just predicts 'negative' for everyone",
                "The dataset is too small",
                "Need more features"
            ],
            correct: 1,
            explanation: "With 90% negative cases, a model that always predicts 'negative' achieves 90% accuracy. The 95% accuracy suggests minimal actual learning. Sensitivity (true positive rate) would be near 0%.",
            insight: "In medical diagnosis, sensitivity (detecting true positives) is often more important than overall accuracy. Always evaluate both sensitivity and specificity, especially with imbalanced datasets.",
            category: "clinical"
        },
        {
            scenario: "⏰ Real-Time Sepsis Prediction",
            question: "Your sepsis prediction model uses the last 6 hours of patient data to predict sepsis in the next 2 hours. How should you validate this?",
            options: [
                "Use expanding window validation",
                "Use rolling window validation with 2-hour gap",
                "Random time-based splits",
                "Bootstrap sampling of time points"
            ],
            correct: 1,
            explanation: "Rolling window with gap prevents using future information and maintains the 6-hour input window structure while respecting the 2-hour prediction horizon.",
            insight: "Clinical prediction models must respect real-time constraints. Validation should mirror deployment: use historical data to predict future outcomes with appropriate time gaps.",
            category: "method"
        },
        {
            scenario: "🧬 Genomics + Clinical Data",
            question: "You have genetic data (unchanged) and clinical data (time-varying) for heart disease prediction. Your AUC jumps from 0.72 to 0.95 when adding one new clinical feature. What should you investigate first?",
            options: [
                "Whether the feature has missing values",
                "If the feature contains information from after the outcome", 
                "The feature's correlation with genetics",
                "Whether you need more data"
            ],
            correct: 1,
            explanation: "A dramatic performance improvement suggests potential data leakage. The new feature might contain information collected after the heart disease diagnosis, making prediction artificially easy.",
            insight: "Suspicious performance jumps often indicate data leakage. In clinical settings, ensure all features represent information available at prediction time, not diagnostic information collected afterward.",
            category: "leakage"
        },
        {
            scenario: "📊 Small Clinical Dataset",
            question: "You have only 150 samples for a rare disease prediction model. Standard 5-fold CV gives highly variable results (AUC: 0.6-0.9 across folds). What's the best approach?",
            options: [
                "Use leave-one-out cross-validation",
                ".632 bootstrap validation",
                "Reduce to 3-fold cross-validation",
                "Use a single train-test split"
            ],
            correct: 1,
            explanation: ".632 bootstrap validation is designed for small datasets and provides more stable estimates than k-fold CV by using multiple bootstrap samples with out-of-bag evaluation.",
            insight: "Small medical datasets require specialized validation approaches. Bootstrap methods can provide more reliable estimates when traditional CV gives unstable results due to limited sample size.",
            category: "method"
        },
        {
            scenario: "🌍 Global Health Study",
            question: "Your malaria prediction model is trained on data from Kenya. You want to validate its performance for deployment in Nigeria. What's the most appropriate validation strategy?",
            options: [
                "10-fold cross-validation on Kenyan data",
                "Leave-one-out validation on Kenyan data",
                "Test on separate Nigerian dataset",
                "Bootstrap validation on combined Kenya-Nigeria data"
            ],
            correct: 2,
            explanation: "External validation on Nigerian data is essential to test geographical generalizability. Different regions have different malaria strains, vector populations, and patient characteristics.",
            insight: "Geographic validation is crucial for global health applications. Population differences, disease variants, and environmental factors can significantly impact model performance across regions.",
            category: "clinical"
        },
        {
            scenario: "💊 Drug Response Prediction",
            question: "You're predicting drug response using patient characteristics and lab values. After data preprocessing (normalization), all folds show nearly identical means and standard deviations. This suggests:",
            options: [
                "Perfect data preprocessing",
                "Data was normalized before splitting (causing leakage)",
                "The dataset is very homogeneous", 
                "You need more diverse data"
            ],
            correct: 1,
            explanation: "Identical statistics across folds indicate global normalization before splitting, which leaks test set information into training. Each fold should be normalized independently.",
            insight: "Preprocessing steps like normalization, scaling, or imputation must be done separately for each CV fold. Global preprocessing before splitting introduces subtle but systematic bias.",
            category: "leakage"
        },
        {
            scenario: "📈 Clinical Decision Support",
            question: "Your clinical decision support model shows 92% accuracy in validation but only 78% in real-world deployment. What's the most likely explanation?",
            options: [
                "Different patient population in deployment",
                "Data leakage during validation",
                "Model overfitting to training data",
                "All of the above are possible"
            ],
            correct: 3,
            explanation: "All factors can cause validation-deployment gaps: population shift, data leakage inflating validation performance, and overfitting. This is why rigorous validation methodology is crucial.",
            insight: "The validation-deployment gap is a critical issue in AI. Proper validation methodology, population matching, and leakage prevention are all essential for reliable clinical performance.",
            category: "clinical"
        },
        {
            scenario: "🫀 Medical Imaging - Radiology",
            question: "You're training a chest X-ray pneumonia detection model using 10,000 images from 2,500 patients (4 images per patient on average). What's the critical validation principle?",
            options: [
                "Split randomly at image level for maximum data",
                "Split at patient level to prevent data leakage",
                "Use stratified sampling by pneumonia presence",
                "Balance images per patient across folds"
            ],
            correct: 1,
            explanation: "Patient-level splitting is essential because images from the same patient share anatomical similarities, lighting conditions, and disease progression patterns that would leak into validation if mixed.",
            insight: "Medical imaging requires patient-level splitting even for seemingly independent images. Shared patient characteristics create dependencies that violate CV independence assumptions.",
            category: "imaging"
        },
        {
            scenario: "🧪 Laboratory Medicine",
            question: "You're predicting lab test abnormalities using a 5-year dataset where test protocols changed significantly in year 3. How should you structure validation?",
            options: [
                "Random 5-fold cross-validation across all years",
                "Temporal split: years 1-3 train, years 4-5 test",
                "Separate validation for pre/post protocol change periods",
                "Bootstrap sampling from each year equally"
            ],
            correct: 2,
            explanation: "Protocol changes represent distribution shifts that require temporal validation. Training on old protocols and testing on new ones tests real-world adaptation challenges.",
            insight: "Medical practices evolve over time. Temporal validation helps assess model robustness to changing protocols, equipment updates, and evolving clinical standards.",
            category: "temporal"
        },
        {
            scenario: "🏥 Emergency Department Triage",
            question: "Your ED triage model uses patient symptoms, vitals, and 'chief_complaint_processed' field. The processed field shows suspiciously high predictive power. What should you verify?",
            options: [
                "Whether the field has sufficient vocabulary coverage",
                "If triage decisions influenced the processed field creation",
                "Whether the field correlates with patient demographics", 
                "If the preprocessing removes stop words correctly"
            ],
            correct: 1,
            explanation: "If triage outcomes influenced how chief complaints were processed (e.g., coding severity), this creates target leakage where the outcome indirectly influences the predictor.",
            insight: "In clinical settings, human decision-makers often influence data collection. Any field that might have been influenced by the target outcome can introduce subtle but systematic leakage.",
            category: "leakage"
        },
        {
            scenario: "🩺 Wearable Health Monitoring",
            question: "Your wearable device collects heart rate data every minute for arrhythmia detection. You have 6 months of data per user. For robust validation, you should:",
            options: [
                "Split randomly within each user's timeline",
                "Use rolling window validation with user-wise splitting",
                "Take every 5th day as validation across all users",
                "Use the last month from each user as test data"
            ],
            correct: 1,
            explanation: "Rolling window validation with user-wise splitting maintains temporal order, prevents user-level leakage, and mimics real deployment where you predict future states from past data.",
            insight: "Wearable data has both user-specific patterns and temporal dependencies. Validation must address both by keeping users separate and maintaining time order within users.",
            category: "temporal"
        },
        {
            scenario: "🧬 Precision Medicine - Pharmacogenomics",
            question: "You're predicting drug metabolism rates using genetic variants and patient demographics. Your model achieves 0.95 AUC, but a simpler model using only age and weight gets 0.91 AUC. This suggests:",
            options: [
                "Genetic data adds meaningful but small improvement",
                "Possible overfitting to genetic noise in small dataset",
                "Perfect—genetics should strongly predict metabolism",
                "Need to include more genetic variants"
            ],
            correct: 1,
            explanation: "Small improvement from genetics despite high individual AUCs suggests overfitting. Complex genetic models can memorize noise in small datasets while simple clinical features capture the main signal.",
            insight: "In precision medicine, sophisticated genetic models may overfit to limited samples. Always compare against simple baselines and validate that genetic complexity adds genuine predictive value.",
            category: "method"
        },
        {
            scenario: "🔬 Pathology - Digital Histology",
            question: "You're developing an AI system to detect cancer in histopathology slides. Your dataset has 10,000 tissue patches from 500 patients (20 patches per patient). What's the most critical validation consideration?",
            options: [
                "Ensuring equal representation of cancer vs. normal patches",
                "Splitting at the patient level to prevent data leakage",
                "Using stratified sampling by tissue type",
                "Maximizing the number of patches in training"
            ],
            correct: 1,
            explanation: "Patient-level splitting is essential because patches from the same patient share biological characteristics, staining patterns, and disease states that would create unrealistic performance if mixed between training and testing.",
            insight: "In digital pathology, multiple patches from the same patient are highly correlated. Patient-level validation ensures the model can generalize to new patients, not just new patches from known patients.",
            category: "imaging"
        },
        {
            scenario: "🏥 Electronic Health Records (EHR)",
            question: "You're building a readmission risk model using EHR data spanning 2015-2023. The hospital implemented a new EHR system in 2019. What validation strategy addresses this challenge?",
            options: [
                "Random 80/20 split across all years",
                "Temporal split: 2015-2018 train, 2019-2023 test", 
                "Stratified split maintaining yearly proportions",
                "Leave-one-year-out cross-validation"
            ],
            correct: 1,
            explanation: "The EHR system change represents a significant distribution shift. Temporal splitting tests whether models trained on the old system can adapt to new data patterns and coding practices.",
            insight: "Healthcare IT changes create natural breakpoints in data. Temporal validation across such transitions is crucial for assessing model robustness to infrastructure changes.",
            category: "temporal"
        },
        {
            scenario: "💉 Clinical Trials - Adverse Event Prediction",
            question: "You're analyzing adverse events across multiple clinical trial sites. Each site has different patient populations and protocols. Your model performs well in CV (AUC=0.85) but poorly at some sites in deployment (AUC=0.65). What's the likely cause?",
            options: [
                "Insufficient training data overall",
                "Site-specific effects not captured in validation",
                "Model complexity is too low",
                "Need more adverse event examples"
            ],
            correct: 1,
            explanation: "Poor generalization across sites suggests site-specific effects (populations, protocols, staff practices) that weren't properly validated. Leave-one-site-out CV would have revealed this issue.",
            insight: "Multi-site clinical data often has hidden site-specific biases. Validation strategies must explicitly test cross-site generalizability to ensure robust deployment.",
            category: "clinical"
        },
        {
            scenario: "🩻 Medical Imaging - MRI Sequences",
            question: "Your brain tumor segmentation model uses multiple MRI sequences (T1, T2, FLAIR). During validation, you notice performance drops significantly when any sequence is missing. This indicates:",
            options: [
                "The model is working correctly—all sequences are needed",
                "Potential overfitting to the complete sequence combinations",
                "Need to collect more data with missing sequences",
                "The model architecture needs modification"
            ],
            correct: 1,
            explanation: "Severe performance degradation with missing sequences suggests overfitting to complete data patterns. Robust AI should gracefully handle common real-world scenarios like missing sequences.",
            insight: "Medical imaging validation should include realistic missing data scenarios. Models that break with incomplete inputs will fail in clinical practice where perfect data is rare.",
            category: "imaging"
        },
        {
            scenario: "🧪 Biomarker Discovery - Metabolomics",
            question: "You're discovering metabolomic biomarkers for early Alzheimer's detection using samples from 200 patients. After feature selection, your model uses 50 metabolites and achieves 95% accuracy. What's concerning?",
            options: [
                "95% accuracy is too high to be realistic",
                "The ratio of features to samples creates overfitting risk",
                "Need to reduce to fewer than 10 metabolites",
                "Should validate on healthy controls only"
            ],
            correct: 1,
            explanation: "With 200 samples and 50 features, the model risks overfitting especially in high-dimensional metabolomics data. Nested CV with proper feature selection is essential to get unbiased performance estimates.",
            insight: "High-dimensional biomarker discovery requires careful validation. When features approach sample size, standard CV can be overly optimistic due to selection bias and overfitting.",
            category: "method"
        }
    ];
    
    // Initialize quiz UI - check if elements exist first
    const totalQuestionsEl = document.getElementById('total-questions');
    if (totalQuestionsEl) {
        totalQuestionsEl.textContent = quizData.length;
    }
    resetQuiz();
}

function resetQuiz() {
    currentQuizQuestion = 0;
    userAnswers = [];
    quizStarted = false;
    
    // Hide results and show quiz - check if elements exist
    const resultsEl = document.getElementById('quiz-results');
    const cardEl = document.getElementById('quiz-card');
    
    if (resultsEl) resultsEl.style.display = 'none';
    if (cardEl) cardEl.style.display = 'block';
    
    // Only load first question if quiz data exists
    if (quizData && quizData.length > 0) {
        loadQuestion(0);
    }
}

function loadQuestion(questionIndex) {
    if (questionIndex >= quizData.length || !quizData || quizData.length === 0) {
        showQuizResults();
        return;
    }
    
    const question = quizData[questionIndex];
    
    // Update question elements - check if they exist
    const scenarioEl = document.getElementById('quiz-scenario');
    const questionEl = document.getElementById('quiz-question');
    const currentEl = document.getElementById('current-question');
    
    if (scenarioEl) scenarioEl.textContent = question.scenario;
    if (questionEl) questionEl.textContent = question.question;
    if (currentEl) currentEl.textContent = questionIndex + 1;
    
    // Create option buttons
    const optionsContainer = document.getElementById('quiz-options');
    if (!optionsContainer) {
        console.error('Quiz options container not found');
        return;
    }
    optionsContainer.innerHTML = '';
    
    question.options.forEach((option, index) => {
        const button = document.createElement('button');
        button.className = 'quiz-option';
        button.textContent = option;
        button.onclick = () => selectQuizOption(index, button);
        optionsContainer.appendChild(button);
    });
    
    // Hide explanation
    document.getElementById('quiz-explanation').style.display = 'none';
    
    // Update buttons
    document.getElementById('quiz-prev').disabled = questionIndex === 0;
    document.getElementById('quiz-submit').style.display = 'none';
    document.getElementById('quiz-next').style.display = 'inline-block';
    
    // Update progress
    const progress = ((questionIndex + 1) / quizData.length) * 100;
    document.getElementById('quiz-progress').style.width = progress + '%';
}

function selectQuizOption(optionIndex, buttonElement) {
    // Clear previous selections
    document.querySelectorAll('.quiz-option').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    // Mark this option as selected
    buttonElement.classList.add('selected');
    
    // Store answer
    userAnswers[currentQuizQuestion] = optionIndex;
    
    // Show submit button
    document.getElementById('quiz-submit').style.display = 'inline-block';
}

function submitAnswer() {
    const question = quizData[currentQuizQuestion];
    const userAnswer = userAnswers[currentQuizQuestion];
    
    // Show explanation
    document.getElementById('explanation-content').textContent = question.explanation;
    document.getElementById('clinical-insight-text').textContent = question.insight;
    document.getElementById('quiz-explanation').style.display = 'block';
    
    // Color code the options
    const options = document.querySelectorAll('.quiz-option');
    options.forEach((option, index) => {
        if (index === question.correct) {
            option.classList.add('correct');
        } else if (index === userAnswer && index !== question.correct) {
            option.classList.add('incorrect');
        }
        option.disabled = true;
    });
    
    // Hide submit button
    document.getElementById('quiz-submit').style.display = 'none';
}

function nextQuestion() {
    currentQuizQuestion++;
    loadQuestion(currentQuizQuestion);
}

function previousQuestion() {
    if (currentQuizQuestion > 0) {
        currentQuizQuestion--;
        loadQuestion(currentQuizQuestion);
    }
}

function showQuizResults() {
    // Don't show results if quiz hasn't been started or no questions answered
    if (!quizStarted || userAnswers.length === 0) {
        console.log('Quiz not properly started or no answers provided');
        return;
    }
    
    // Calculate score
    let score = 0;
    let leakageScore = 0, methodScore = 0, clinicalScore = 0, imagingScore = 0, temporalScore = 0;
    let leakageTotal = 0, methodTotal = 0, clinicalTotal = 0, imagingTotal = 0, temporalTotal = 0;
    
    quizData.forEach((question, index) => {
        const correct = userAnswers[index] === question.correct;
        if (correct) score++;
        
        // Category scoring
        if (question.category === 'leakage') {
            leakageTotal++;
            if (correct) leakageScore++;
        } else if (question.category === 'method') {
            methodTotal++;
            if (correct) methodScore++;
        } else if (question.category === 'clinical') {
            clinicalTotal++;
            if (correct) clinicalScore++;
        } else if (question.category === 'imaging') {
            imagingTotal++;
            if (correct) imagingScore++;
        } else if (question.category === 'temporal') {
            temporalTotal++;
            if (correct) temporalScore++;
        }
    });
    
    // Update displays
    document.getElementById('final-score').textContent = score;
    
    // Update skill bars - check for elements and handle division by zero
    const leakagePercent = leakageTotal > 0 ? (leakageScore / leakageTotal) * 100 : 0;
    const methodPercent = methodTotal > 0 ? (methodScore / methodTotal) * 100 : 0;
    const clinicalPercent = clinicalTotal > 0 ? (clinicalScore / clinicalTotal) * 100 : 0;
    const imagingPercent = imagingTotal > 0 ? (imagingScore / imagingTotal) * 100 : 0;
    const temporalPercent = temporalTotal > 0 ? (temporalScore / temporalTotal) * 100 : 0;
    
    const leakageSkillEl = document.getElementById('leakage-skill');
    const methodSkillEl = document.getElementById('method-skill');
    const clinicalSkillEl = document.getElementById('clinical-skill');
    const imagingSkillEl = document.getElementById('imaging-skill');
    const temporalSkillEl = document.getElementById('temporal-skill');
    
    const leakageScoreEl = document.getElementById('leakage-score');
    const methodScoreEl = document.getElementById('method-score');
    const clinicalScoreEl = document.getElementById('clinical-score');
    const imagingScoreEl = document.getElementById('imaging-score');
    const temporalScoreEl = document.getElementById('temporal-score');
    
    if (leakageSkillEl) leakageSkillEl.style.width = leakagePercent + '%';
    if (methodSkillEl) methodSkillEl.style.width = methodPercent + '%';
    if (clinicalSkillEl) clinicalSkillEl.style.width = clinicalPercent + '%';
    if (imagingSkillEl) imagingSkillEl.style.width = imagingPercent + '%';
    if (temporalSkillEl) temporalSkillEl.style.width = temporalPercent + '%';
    
    if (leakageScoreEl) leakageScoreEl.textContent = `${leakageScore}/${leakageTotal}`;
    if (methodScoreEl) methodScoreEl.textContent = `${methodScore}/${methodTotal}`;
    if (clinicalScoreEl) clinicalScoreEl.textContent = `${clinicalScore}/${clinicalTotal}`;
    if (imagingScoreEl) imagingScoreEl.textContent = `${imagingScore}/${imagingTotal}`;
    if (temporalScoreEl) temporalScoreEl.textContent = `${temporalScore}/${temporalTotal}`;
    
    // Generate recommendations
    const recommendations = generateLearningRecommendations(leakagePercent, methodPercent, clinicalPercent, imagingPercent, temporalPercent);
    document.getElementById('learning-recommendations').innerHTML = recommendations;
    
    // Show results
    const cardEl = document.getElementById('quiz-card');
    const resultsEl = document.getElementById('quiz-results');
    
    if (cardEl) cardEl.style.display = 'none';
    if (resultsEl) resultsEl.style.display = 'block';
}

function generateLearningRecommendations(leakage, method, clinical, imaging, temporal) {
    let recommendations = '<h4>📚 Personalized Learning Recommendations:</h4><ul>';
    
    if (leakage < 70) {
        recommendations += '<li>📖 Review our Data Leakage notebook to understand temporal and feature leakage</li>';
    }
    if (method < 70) {
        recommendations += '<li>🔧 Practice with our CV Method Comparison demos to understand when to use each approach</li>';
    }
    if (clinical < 70) {
        recommendations += '<li>🏥 Study our Clinical Application examples to see real-world medical validation scenarios</li>';
    }
    if (imaging < 70) {
        recommendations += '<li>🫀 Explore our Medical Imaging notebooks to master patient-level splitting techniques</li>';
    }
    if (temporal < 70) {
        recommendations += '<li>⏰ Practice with our Temporal Validation demos to understand time-series cross-validation</li>';
    }
    
    const avgScore = (leakage + method + clinical + imaging + temporal) / 5;
    if (avgScore >= 80) {
        recommendations += '<li>🌟 Excellent work! You have a strong understanding of medical CV. Consider contributing to the trustcv project!</li>';
    } else if (avgScore >= 70) {
        recommendations += '<li>🎯 Good progress! Focus on your weaker areas and review the comprehensive CV methods table</li>';
    }
    
    recommendations += '</ul>';
    return recommendations;
}

function restartQuiz() {
    resetQuiz();
}

// Start Quiz Function (missing from original)
function startQuiz() {
    // Make sure quiz data is initialized first
    if (!quizData || quizData.length === 0) {
        console.log('Initializing quiz data...');
        initializeQuiz();
    }
    
    // Hide intro sections
    const introEl = document.querySelector('.quiz-intro');
    const featuresEl = document.querySelector('.quiz-features');
    const quizEl = document.getElementById('interactive-quiz');
    
    if (introEl) introEl.style.display = 'none';
    if (featuresEl) featuresEl.style.display = 'none';
    if (quizEl) quizEl.style.display = 'block';
    
    // Initialize and start quiz
    quizStarted = true;
    currentQuizQuestion = 0;
    userAnswers = [];
    
    // Load first question
    if (quizData && quizData.length > 0) {
        loadQuestion(0);
    } else {
        console.error('Quiz data not available');
    }
}

// Medical Imaging Demo
function runImagingDemo() {
    const strategy = document.getElementById('imaging-strategy').value;
    const handleImbalance = document.getElementById('class-imbalance').checked;
    
    // Simulate different validation strategies
    let performance, patientAuc, imageAuc;
    
    if (strategy === 'image-level') {
        // Overoptimistic due to patient-level leakage
        imageAuc = (0.92 + Math.random() * 0.05).toFixed(3);
        patientAuc = (0.73 + Math.random() * 0.08).toFixed(3);
        performance = {
            reported: parseFloat(imageAuc),
            actual: parseFloat(patientAuc),
            gap: ((parseFloat(imageAuc) - parseFloat(patientAuc)) * 100).toFixed(1)
        };
    } else {
        // Realistic performance with proper patient-level splitting
        const baseAuc = strategy === 'stratified-patient' ? 0.76 : 0.74;
        const adjustedAuc = handleImbalance ? baseAuc + 0.02 : baseAuc;
        patientAuc = (adjustedAuc + Math.random() * 0.06).toFixed(3);
        performance = {
            reported: parseFloat(patientAuc),
            actual: parseFloat(patientAuc),
            gap: 0
        };
    }
    
    createImagingVisualization(strategy, performance);
    updateImagingExplanation(strategy, performance, handleImbalance);
}

function createImagingVisualization(strategy, performance) {
    const strategies = ['Image-Level Split', 'Patient-Level Split', 'Stratified Patient'];
    const aucs = [0.92, 0.74, 0.76];
    const actualPerf = [0.73, 0.74, 0.76]; // Real-world performance
    
    const currentIndex = ['image-level', 'patient-level', 'stratified-patient'].indexOf(strategy);
    const colors = strategies.map((_, i) => i === currentIndex ? '#f44336' : i === 0 ? '#ffcccb' : '#4caf50');
    
    const trace1 = {
        x: strategies,
        y: aucs,
        name: 'Validation AUC',
        type: 'bar',
        marker: { color: colors }
    };
    
    const trace2 = {
        x: strategies,
        y: actualPerf,
        name: 'Real-World Performance',
        type: 'scatter',
        mode: 'markers',
        marker: { 
            color: 'black',
            size: 10,
            symbol: 'diamond'
        }
    };
    
    const layout = {
        title: 'Medical Imaging CV: Validation vs Real-World Performance',
        yaxis: { title: 'AUC Score', range: [0.6, 1.0] },
        barmode: 'group',
        annotations: [{
            x: 0,
            y: 0.85,
            text: '⚠️ Overoptimistic!',
            showarrow: true,
            arrowcolor: '#f44336',
            font: { color: '#f44336', size: 12 }
        }]
    };
    
    Plotly.newPlot('imaging-demo-plot', [trace1, trace2], layout, {responsive: true});
}

function updateImagingExplanation(strategy, performance, handleImbalance) {
    const explanations = {
        'image-level': `❌ PROBLEMATIC: Images from same patients appear in both training and test sets. Model learns patient-specific features rather than lesion characteristics.`,
        'patient-level': `✅ CORRECT: All images from each patient stay together. Model must generalize to new patients, matching clinical deployment.`,
        'stratified-patient': `✅ OPTIMAL: Patient-level splitting while maintaining melanoma/benign balance across folds.`
    };
    
    const deploymentGap = performance.gap > 15 ? 'SEVERE' : performance.gap > 5 ? 'MODERATE' : 'MINIMAL';
    const gapColor = performance.gap > 15 ? '#f44336' : performance.gap > 5 ? '#ff9800' : '#4caf50';
    
    document.getElementById('imaging-explanation').innerHTML = `
        <h4>Strategy: ${strategy.replace('-', ' ').toUpperCase()}</h4>
        <p><strong>Validation Performance:</strong> ${performance.reported.toFixed(3)} AUC</p>
        <p><strong>Validation-Deployment Gap:</strong> <span style="color: ${gapColor}">${deploymentGap}</span> (${performance.gap}% difference)</p>
        <p><strong>Explanation:</strong> ${explanations[strategy]}</p>
        ${handleImbalance ? '<p><strong>Class Balance:</strong> ✅ Handling 1:4 melanoma ratio properly</p>' : ''}
        <div class="dermatology-insight">
            <h5>🏥 Dermatology Deployment:</h5>
            <p>${strategy === 'image-level' ? 
                'This validation approach would likely fail FDA review due to inflated performance estimates. Real-world deployment would show significantly worse performance.' :
                'This approach properly tests generalization to new patients and would be acceptable for regulatory submission and clinical deployment.'}</p>
        </div>
    `;
}

// Genomics Demo
function runGenomicsDemo() {
    const approach = document.getElementById('genomics-approach').value;
    const featureSelection = document.getElementById('feature-selection').value;
    
    // Simulate different validation approaches for genomics data
    const results = simulateGenomicsValidation(approach, featureSelection);
    createGenomicsVisualization(approach, results);
    updateGenomicsExplanation(approach, results, featureSelection);
}

function simulateGenomicsValidation(approach, featureSelection) {
    let baseAuc, stability, batchEffect;
    
    switch(approach) {
        case 'standard':
            baseAuc = 0.88;
            stability = 0.12;
            batchEffect = 'High';
            break;
        case 'stratified':
            baseAuc = 0.82;
            stability = 0.08;
            batchEffect = 'Medium';
            break;
        case 'batch-aware':
            baseAuc = 0.76;
            stability = 0.06;
            batchEffect = 'Low';
            break;
        case 'leave-ancestry-out':
            baseAuc = 0.71;
            stability = 0.15;
            batchEffect = 'Very Low';
            break;
    }
    
    // Adjust for feature selection
    const fsAdjustment = {
        'none': 0,
        'univariate': -0.02,
        'lasso': 0.01,
        'stability': 0.03
    };
    
    baseAuc += fsAdjustment[featureSelection];
    
    return {
        auc: baseAuc.toFixed(3),
        stability: stability.toFixed(3),
        batchEffect,
        approach,
        featureSelection
    };
}

function createGenomicsVisualization(approach, results) {
    const approaches = ['Standard K-Fold', 'Ancestry-Stratified', 'Batch-Aware', 'Leave-Population-Out'];
    const aucs = [0.88, 0.82, 0.76, 0.71];
    const stabilities = [0.12, 0.08, 0.06, 0.15];
    
    const currentIndex = ['standard', 'stratified', 'batch-aware', 'leave-ancestry-out'].indexOf(approach);
    const colors = approaches.map((_, i) => i === currentIndex ? '#9c27b0' : '#e1bee7');
    
    const trace1 = {
        x: approaches,
        y: aucs,
        name: 'Mean AUC',
        type: 'bar',
        marker: { color: colors },
        error_y: {
            type: 'data',
            array: stabilities,
            visible: true
        }
    };
    
    const layout = {
        title: 'Genomics CV: Population Stratification Impact',
        yaxis: { title: 'AUC Score', range: [0.5, 1.0] },
        xaxis: { tickangle: -45 },
        annotations: [{
            x: 0,
            y: 0.95,
            text: 'Likely Overfitting',
            showarrow: true,
            arrowcolor: '#f44336',
            font: { color: '#f44336' }
        }]
    };
    
    Plotly.newPlot('genomics-demo-plot', [trace1], layout, {responsive: true});
}

function updateGenomicsExplanation(approach, results, featureSelection) {
    const explanations = {
        'standard': `Standard CV ignores genetic relatedness and population structure, leading to inflated estimates.`,
        'stratified': `Ancestry stratification maintains population balance but may not fully address batch effects.`,
        'batch-aware': `Batch-aware validation accounts for technical artifacts from different sequencing runs.`,
        'leave-ancestry-out': `Most rigorous test of generalization across different populations, essential for global deployment.`
    };
    
    const batchImpact = results.batchEffect === 'High' ? '⚠️ HIGH RISK' : 
                       results.batchEffect === 'Medium' ? '⚠️ MODERATE RISK' : '✅ LOW RISK';
    
    document.getElementById('genomics-explanation').innerHTML = `
        <h4>Approach: ${approach.replace('-', ' ').toUpperCase()}</h4>
        <p><strong>Performance:</strong> ${results.auc} ± ${results.stability} AUC</p>
        <p><strong>Batch Effect Risk:</strong> ${batchImpact}</p>
        <p><strong>Feature Selection:</strong> ${featureSelection.toUpperCase()}</p>
        <p><strong>Explanation:</strong> ${explanations[approach]}</p>
        <div class="genomics-insight">
            <h5>🧬 Genomic Medicine Insight:</h5>
            <p>${approach === 'leave-ancestry-out' ? 
                'This approach tests true generalization across populations and would be required for global deployment of genomic medicine tools.' :
                approach === 'standard' ? 
                'This approach risks learning population-specific genetic patterns rather than disease-relevant variants.' :
                'This approach provides a reasonable balance between validation rigor and computational efficiency.'}</p>
        </div>
    `;
}

// Drug Discovery Demo  
function runDrugDemo() {
    const method = document.getElementById('drug-cv-method').value;
    const threshold = document.getElementById('similarity-threshold').value;
    
    // Update threshold display
    document.getElementById('similarity-value').textContent = threshold;
    
    const results = simulateDrugDiscoveryCV(method, threshold);
    createDrugVisualization(method, results);
    updateDrugExplanation(method, results, threshold);
}

function simulateDrugDiscoveryCV(method, threshold) {
    let validationR2, deploymentR2, scaffoldNovelty;
    
    switch(method) {
        case 'random':
            validationR2 = 0.76;
            deploymentR2 = 0.48; // Large drop due to scaffold similarity
            scaffoldNovelty = 'Low';
            break;
        case 'scaffold':
            validationR2 = 0.63;
            deploymentR2 = 0.61; // Similar performance on novel scaffolds
            scaffoldNovelty = 'High';
            break;
        case 'temporal':
            validationR2 = 0.68;
            deploymentR2 = 0.65; // Accounts for discovery timeline
            scaffoldNovelty = 'Medium';
            break;
        case 'target-based':
            validationR2 = 0.71;
            deploymentR2 = 0.68; // Good generalization across targets
            scaffoldNovelty = 'Medium';
            break;
    }
    
    // Adjust based on similarity threshold
    const thresholdEffect = (parseFloat(threshold) - 0.6) * 0.1;
    validationR2 -= thresholdEffect;
    
    return {
        validationR2: validationR2.toFixed(3),
        deploymentR2: deploymentR2.toFixed(3),
        gap: ((validationR2 - deploymentR2) * 100).toFixed(1),
        scaffoldNovelty,
        method
    };
}

function createDrugVisualization(method, results) {
    const methods = ['Random Split', 'Scaffold Split', 'Temporal Split', 'Target-Based'];
    const validation = [0.76, 0.63, 0.68, 0.71];
    const deployment = [0.48, 0.61, 0.65, 0.68];
    
    const currentIndex = ['random', 'scaffold', 'temporal', 'target-based'].indexOf(method);
    
    const trace1 = {
        x: methods,
        y: validation,
        name: 'Validation R²',
        type: 'bar',
        marker: { color: methods.map((_, i) => i === currentIndex ? '#ff9800' : '#ffcc80') }
    };
    
    const trace2 = {
        x: methods,
        y: deployment,
        name: 'Novel Compounds R²',
        type: 'scatter',
        mode: 'markers',
        marker: { 
            color: 'black',
            size: 12,
            symbol: 'diamond'
        }
    };
    
    const layout = {
        title: 'Drug Discovery CV: Validation vs Novel Compound Performance',
        yaxis: { title: 'R² Score', range: [0.3, 0.8] },
        xaxis: { tickangle: -45 },
        barmode: 'group'
    };
    
    Plotly.newPlot('drug-discovery-plot', [trace1, trace2], layout, {responsive: true});
}

function updateDrugExplanation(method, results, threshold) {
    const explanations = {
        'random': `Random splitting allows similar molecular scaffolds in both training and test sets, creating unrealistic performance estimates.`,
        'scaffold': `Scaffold splitting ensures test compounds have novel chemical structures, matching real drug discovery scenarios.`,
        'temporal': `Temporal splitting mimics the discovery timeline, where newer compounds are predicted using historical data.`,
        'target-based': `Target-based CV tests generalization across different protein targets and binding sites.`
    };
    
    const deploymentReliability = parseFloat(results.gap) < 5 ? 'HIGH' : 
                                 parseFloat(results.gap) < 15 ? 'MODERATE' : 'LOW';
    const reliabilityColor = deploymentReliability === 'HIGH' ? '#4caf50' : 
                            deploymentReliability === 'MODERATE' ? '#ff9800' : '#f44336';
    
    document.getElementById('drug-explanation').innerHTML = `
        <h4>Method: ${method.replace('-', ' ').toUpperCase()}</h4>
        <p><strong>Validation R²:</strong> ${results.validationR2}</p>
        <p><strong>Novel Compound Performance:</strong> ${results.deploymentR2}</p>
        <p><strong>Performance Gap:</strong> ${results.gap}%</p>
        <p><strong>Deployment Reliability:</strong> <span style="color: ${reliabilityColor}">${deploymentReliability}</span></p>
        <p><strong>Scaffold Novelty:</strong> ${results.scaffoldNovelty}</p>
        <p><strong>Similarity Threshold:</strong> ${threshold}</p>
        <p><strong>Explanation:</strong> ${explanations[method]}</p>
        <div class="pharma-insight">
            <h5>💊 Pharmaceutical Impact:</h5>
            <p>${method === 'scaffold' ? 
                'Scaffold splitting is the gold standard for drug discovery CV, ensuring models work on truly novel chemical matter.' :
                method === 'random' ?
                'Random splitting gives false confidence and would lead to poor performance on innovative drug candidates.' :
                'This approach provides a reasonable balance for specific drug discovery applications.'}</p>
        </div>
    `;
}

// Epidemiology Demo
function runEpiDemo() {
    const cvType = document.getElementById('epi-cv-type').value;
    const window = document.getElementById('prediction-window').value;
    
    // Update window display
    document.getElementById('window-value').textContent = window;
    
    const results = simulateEpidemiologyCV(cvType, window);
    createEpiVisualization(cvType, results);
    updateEpiExplanation(cvType, results, window);
}

function simulateEpidemiologyCV(cvType, window) {
    let sensitivity, specificity, spatialGeneralization;
    
    const windowEffect = (parseInt(window) - 14) * 0.01; // Longer windows = harder prediction
    
    switch(cvType) {
        case 'standard':
            sensitivity = Math.max(0.5, 0.82 + windowEffect);
            specificity = Math.max(0.6, 0.87 + windowEffect);
            spatialGeneralization = 'Poor';
            break;
        case 'spatial-block':
            sensitivity = Math.max(0.45, 0.74 + windowEffect);
            specificity = Math.max(0.55, 0.81 + windowEffect);
            spatialGeneralization = 'Good';
            break;
        case 'leave-country-out':
            sensitivity = Math.max(0.4, 0.68 + windowEffect);
            specificity = Math.max(0.5, 0.76 + windowEffect);
            spatialGeneralization = 'Excellent';
            break;
        case 'spatiotemporal':
            sensitivity = Math.max(0.42, 0.71 + windowEffect);
            specificity = Math.max(0.52, 0.78 + windowEffect);
            spatialGeneralization = 'Very Good';
            break;
    }
    
    return {
        sensitivity: sensitivity.toFixed(3),
        specificity: specificity.toFixed(3),
        spatialGeneralization,
        cvType
    };
}

function createEpiVisualization(cvType, results) {
    const methods = ['Standard TS CV', 'Spatial Block', 'Leave-Country-Out', 'Spatiotemporal'];
    const sensitivities = [0.82, 0.74, 0.68, 0.71];
    const specificities = [0.87, 0.81, 0.76, 0.78];
    
    const currentIndex = ['standard', 'spatial-block', 'leave-country-out', 'spatiotemporal'].indexOf(cvType);
    
    const trace1 = {
        x: sensitivities,
        y: specificities,
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: methods.map((_, i) => i === currentIndex ? 15 : 8),
            color: methods.map((_, i) => i === currentIndex ? '#2196f3' : '#90caf9'),
            line: { color: 'black', width: 1 }
        },
        text: methods,
        textposition: 'top center',
        name: 'CV Methods'
    };
    
    const layout = {
        title: 'Epidemiology CV: Sensitivity vs Specificity',
        xaxis: { title: 'Sensitivity (Outbreak Detection Rate)', range: [0.3, 0.9] },
        yaxis: { title: 'Specificity (True Negative Rate)', range: [0.4, 0.95] },
        annotations: [{
            x: 0.82,
            y: 0.87,
            text: 'Optimistic',
            showarrow: true,
            arrowcolor: '#ff9800'
        }, {
            x: 0.68,
            y: 0.76,
            text: 'Realistic',
            showarrow: true,
            arrowcolor: '#4caf50'
        }]
    };
    
    Plotly.newPlot('epidemiology-plot', [trace1], layout, {responsive: true});
}

function updateEpiExplanation(cvType, results, window) {
    const explanations = {
        'standard': `Standard time series CV ignores spatial dependencies and may overestimate performance in new regions.`,
        'spatial-block': `Spatial block CV tests performance in geographically distant regions, accounting for local transmission patterns.`,
        'leave-country-out': `Most rigorous approach for testing international deployment of outbreak detection systems.`,
        'spatiotemporal': `Combines spatial and temporal constraints, providing realistic estimates for real-time surveillance systems.`
    };
    
    const deploymentRisk = results.spatialGeneralization === 'Poor' ? 'HIGH RISK' :
                          results.spatialGeneralization === 'Good' ? 'MODERATE RISK' : 'LOW RISK';
    const riskColor = deploymentRisk === 'HIGH RISK' ? '#f44336' : 
                     deploymentRisk === 'MODERATE RISK' ? '#ff9800' : '#4caf50';
    
    document.getElementById('epi-explanation').innerHTML = `
        <h4>Strategy: ${cvType.replace('-', ' ').toUpperCase()}</h4>
        <p><strong>Outbreak Detection (Sensitivity):</strong> ${results.sensitivity}</p>
        <p><strong>False Alarm Control (Specificity):</strong> ${results.specificity}</p>
        <p><strong>Prediction Window:</strong> ${window} days ahead</p>
        <p><strong>Geographic Generalization:</strong> ${results.spatialGeneralization}</p>
        <p><strong>International Deployment Risk:</strong> <span style="color: ${riskColor}">${deploymentRisk}</span></p>
        <p><strong>Explanation:</strong> ${explanations[cvType]}</p>
        <div class="public-health-insight">
            <h5>🌍 Public Health Impact:</h5>
            <p>${cvType === 'leave-country-out' ? 
                'This rigorous validation approach ensures outbreak detection models work reliably across different countries and healthcare systems, critical for global health security.' :
                cvType === 'standard' ?
                'Standard validation may fail in new regions due to different population densities, mobility patterns, and healthcare infrastructures.' :
                'This approach provides a good balance between validation rigor and practical deployment considerations.'}</p>
        </div>
    `;
}