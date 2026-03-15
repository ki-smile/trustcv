/**
 * trustcv Interactive Visualizations
 * Main JavaScript file for website interactions
 */

// KI Color Palette - dynamic for theme support
function getColors() {
    if (typeof getThemeColors === 'function') return getThemeColors();
    // Fallback if theme.js hasn't loaded
    return {
        plum: '#870052', darkPlum: '#4F0433', orange: '#FF876F',
        lightBlue: '#EDF4F4', lightOrange: '#FEEEEB', grey: '#6B6B6B',
        text: '#000000', textMuted: '#666666', background: '#FEEEEB',
        plotBg: '#FFFFFF', paperBg: '#EDF4F4', train: '#3498DB', test: '#E74C3C',
        inactive: '#BDC3C7', gridColor: 'rgba(0,0,0,0.06)', axisColor: '#666666',
        titleColor: '#4F0433', annotationColor: '#666666', border: '#EDDBE4',
        surface: '#FFFFFF', cardBg: '#EDF4F4'
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initHeroVisualization();
    initCVVisualization();
    setupMethodSelector();
    setupCodeBlocks();
});

/**
 * Add "Copy" buttons to all code blocks
 */
function setupCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach((block) => {
        // Only add if it's a code block and not already processed
        if (block.querySelector('code') && !block.querySelector('.copy-button')) {
            const button = document.createElement('button');
            button.className = 'copy-button';
            button.innerHTML = '<span class="material-icons">content_copy</span>';
            button.title = 'Copy to clipboard';
            
            button.addEventListener('click', async () => {
                const code = block.querySelector('code').innerText;
                try {
                    await navigator.clipboard.writeText(code);
                    button.innerHTML = '<span class="material-icons">check</span>';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.innerHTML = '<span class="material-icons">content_copy</span>';
                        button.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy!', err);
                }
            });
            
            block.style.position = 'relative';
            block.appendChild(button);
        }
    });
}

/**
 * Hero Section - Animated CV Demo
 */
function initHeroVisualization() {
    var colors = getColors();
    const data = generateSampleData(100);
    
    const trace1 = {
        x: data.x,
        y: data.y,
        mode: 'markers',
        type: 'scatter',
        name: 'Training Data',
        marker: {
            color: colors.plum,
            size: 8,
            opacity: 0.6
        }
    };
    
    const layout = {
        title: 'Cross-Validation in Action',
        xaxis: { title: 'Feature 1', color: colors.axisColor, gridcolor: colors.gridColor },
        yaxis: { title: 'Feature 2', color: colors.axisColor, gridcolor: colors.gridColor },
        showlegend: true,
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { family: 'Roboto, sans-serif', color: colors.text }
    };
    
    Plotly.newPlot('cv-demo-plot', [trace1], layout, {responsive: true});
    
    // Animate fold splitting
    animateFolds();
}

/**
 * Generate sample medical data
 */
function generateSampleData(n) {
    const data = { x: [], y: [], class: [] };
    
    for (let i = 0; i < n; i++) {
        const classLabel = Math.random() > 0.5 ? 1 : 0;
        const xCenter = classLabel ? 3 : 1;
        const yCenter = classLabel ? 3 : 1;
        
        data.x.push(xCenter + (Math.random() - 0.5) * 2);
        data.y.push(yCenter + (Math.random() - 0.5) * 2);
        data.class.push(classLabel);
    }
    
    return data;
}

/**
 * Animate k-fold splitting
 */
function animateFolds() {
    const nSplits = 5;
    let currentFold = 0;
    
    setInterval(() => {
        var colors = getColors();
        const data = generateSampleData(100);
        const foldSize = Math.floor(data.x.length / nSplits);
        const testStart = currentFold * foldSize;
        const testEnd = testStart + foldSize;

        const colors_array = data.x.map((_, idx) => {
            if (idx >= testStart && idx < testEnd) {
                return colors.orange;  // Test fold
            }
            return colors.plum;  // Training folds
        });
        
        const sizes = data.x.map((_, idx) => {
            if (idx >= testStart && idx < testEnd) {
                return 12;  // Larger for test fold
            }
            return 8;
        });
        
        Plotly.restyle('cv-demo-plot', {
            'marker.color': [colors_array],
            'marker.size': [sizes]
        });
        
        currentFold = (currentFold + 1) % nSplits;
    }, 2000);
}

/**
 * Main CV Visualization
 */
function initCVVisualization() {
    updateVisualization();
}

function updateVisualization() {
    const method = document.getElementById('cv-method').value;
    const nSplits = document.getElementById('n-splits').value;
    
    const splitsLabel = document.getElementById('splits-label');
    if (splitsLabel) {
        splitsLabel.textContent = `${nSplits} splits`;
    }
    
    // Clear previous visualization
    const vizContainer = document.getElementById('cv-visualization');
    if (!vizContainer) return;
    vizContainer.innerHTML = '';
    
    // Create visualization based on method
    switch(method) {
        // I.I.D. Methods
        case 'holdout':
            visualizeHoldOut();
            break;
        case 'kfold':
            visualizeKFold(nSplits);
            break;
        case 'stratified':
            visualizeStratified(nSplits);
            break;
        case 'repeated':
            visualizeRepeatedKFold(nSplits);
            break;
        case 'loocv':
            visualizeLOOCV();
            break;
        case 'lpocv':
            visualizeLPOCV();
            break;
        case 'bootstrap':
            visualizeBootstrap();
            break;
        case 'montecarlo':
            visualizeMonteCarlo();
            break;
        case 'nested':
            visualizeNested();
            break;
            
        // Temporal Methods
        case 'temporal':
            visualizeTemporal(nSplits);
            break;
        case 'rolling':
            visualizeRollingWindow();
            break;
        case 'expanding':
            visualizeExpandingWindow();
            break;
        case 'blocked':
            visualizeBlockedTimeSeries(nSplits);
            break;
        case 'purged':
            visualizePurgedKFold(nSplits);
            break;
        case 'cpcv':
            visualizeCPCV();
            break;
        case 'purged_group':
            visualizePurgedGroupTimeSeries();
            break;
        case 'nested_temporal':
            visualizeNestedTemporal();
            break;
            
        // Grouped Methods
        case 'grouped':
            visualizeGrouped(nSplits);
            break;
        case 'stratified_grouped':
            visualizeStratifiedGrouped(nSplits);
            break;
        case 'logo':
            visualizeLOGO();
            break;
        case 'lpgo':
            visualizeLPGO();
            break;
        case 'repeated_grouped':
            visualizeRepeatedGrouped(nSplits);
            break;
        case 'hierarchical':
            visualizeHierarchical();
            break;
        case 'multilevel':
            visualizeMultilevel();
            break;
        case 'nested_grouped':
            visualizeNestedGrouped();
            break;
            
        // Spatial Methods
        case 'spatial_block':
            visualizeSpatialBlock(nSplits);
            break;
        case 'buffered_spatial':
            visualizeBufferedSpatial(nSplits);
            break;
        case 'spatiotemporal':
            visualizeSpatiotemporal();
            break;
        case 'environmental':
            visualizeEnvironmental();
            break;
            
        default:
            visualizeKFold(nSplits);
    }
}

/**
 * Visualize K-Fold CV
 */
function visualizeKFold(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');
    const nSamples = 50;
    
    // Create fold visualization
    const foldDiv = document.createElement('div');
    foldDiv.className = 'fold-visualization';
    
    for (let fold = 0; fold < nSplits; fold++) {
        const foldContainer = document.createElement('div');
        foldContainer.className = 'fold-container';
        foldContainer.innerHTML = `<h4>Fold ${fold + 1}</h4>`;
        
        const samplesDiv = document.createElement('div');
        samplesDiv.className = 'samples-container';
        
        const foldSize = Math.floor(nSamples / nSplits);
        const testStart = fold * foldSize;
        const testEnd = testStart + foldSize;
        
        for (let i = 0; i < nSamples; i++) {
            const sample = document.createElement('div');
            sample.className = 'sample';
            
            if (i >= testStart && i < testEnd) {
                sample.style.backgroundColor = colors.orange;
                sample.title = 'Test';
            } else {
                sample.style.backgroundColor = colors.plum;
                sample.title = 'Train';
            }
            
            samplesDiv.appendChild(sample);
        }
        
        foldContainer.appendChild(samplesDiv);
        foldDiv.appendChild(foldContainer);
    }
    
    container.appendChild(foldDiv);
    
    // Add description
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>K-Fold Cross-Validation</h3>
        <p>Data is divided into ${nSplits} equal folds. Each fold serves as test set once.</p>
        <ul>
            <li>✓ All data used for training and testing</li>
            <li>✓ Reduced variance in performance estimate</li>
            <li>✗ May not preserve class distribution</li>
            <li>✗ Not suitable for grouped or temporal data</li>
        </ul>
    `;
    container.appendChild(description);
}

/**
 * Visualize Stratified K-Fold
 */
function visualizeStratified(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Create interactive plot showing class preservation
    const data = [];
    
    for (let fold = 0; fold < nSplits; fold++) {
        data.push({
            x: ['Class 0', 'Class 1'],
            y: [30, 20],  // Preserved ratio
            name: `Fold ${fold + 1}`,
            type: 'bar',
            marker: {
                color: fold === 0 ? colors.orange : colors.plum
            }
        });
    }
    
    const layout = {
        title: 'Stratified K-Fold: Class Distribution Preserved',
        barmode: 'group',
        xaxis: { title: 'Class', color: colors.axisColor, gridcolor: colors.gridColor },
        yaxis: { title: 'Number of Samples', color: colors.axisColor, gridcolor: colors.gridColor },
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    // Add description
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Stratified K-Fold Cross-Validation</h3>
        <p>Preserves the class distribution in each fold - critical for imbalanced medical datasets.</p>
        <ul>
            <li>✓ Maintains class proportions</li>
            <li>✓ Better for imbalanced datasets</li>
            <li>✓ More reliable performance estimates</li>
            <li>✗ Still not suitable for grouped data</li>
        </ul>
    `;
    container.appendChild(description);
}

/**
 * Visualize Grouped K-Fold
 */
function visualizeGrouped(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Create patient grouping visualization
    const patients = generatePatientData();
    
    const traces = [];
    const patientGroups = 10;
    const colors_palette = [colors.plum, colors.orange, '#4CAF50', '#2196F3', '#FF9800'];
    
    for (let fold = 0; fold < nSplits; fold++) {
        const foldPatients = [];
        const foldSamples = [];
        
        for (let p = fold; p < patientGroups; p += nSplits) {
            foldPatients.push(`Patient ${p + 1}`);
            foldSamples.push(Math.floor(Math.random() * 5) + 3);
        }
        
        traces.push({
            x: foldPatients,
            y: foldSamples,
            name: `Fold ${fold + 1}`,
            type: 'bar',
            marker: { color: colors_palette[fold] }
        });
    }
    
    const layout = {
        title: 'Patient-Grouped K-Fold: No Patient Appears in Multiple Folds',
        barmode: 'stack',
        xaxis: { title: 'Patient ID', color: colors.axisColor, gridcolor: colors.gridColor },
        yaxis: { title: 'Number of Records', color: colors.axisColor, gridcolor: colors.gridColor },
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    // Add description
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Patient-Grouped K-Fold Cross-Validation</h3>
        <p>Essential for medical data: ensures all records from a patient stay together.</p>
        <ul>
            <li>✓ Prevents patient data leakage</li>
            <li>✓ More realistic performance estimate</li>
            <li>✓ Required for longitudinal studies</li>
            <li>⚠️ May result in uneven fold sizes</li>
        </ul>
    `;
    container.appendChild(description);
}

/**
 * Visualize Temporal Split
 */
function visualizeTemporal(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Create time series visualization
    const dates = [];
    const values = [];
    const n = 100;
    
    for (let i = 0; i < n; i++) {
        dates.push(new Date(2024, 0, i + 1));
        values.push(Math.sin(i / 10) * 20 + 50 + Math.random() * 10);
    }
    
    // Create fold markers
    const shapes = [];
    const annotations = [];
    const foldSize = Math.floor(n / nSplits);
    
    for (let fold = 0; fold < nSplits; fold++) {
        const start = fold * foldSize;
        const end = start + foldSize;
        
        shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: dates[start],
            x1: dates[Math.min(end, n - 1)],
            y0: 0,
            y1: 1,
            fillcolor: fold === nSplits - 1 ? colors.orange : colors.plum,
            opacity: 0.2,
            line: { width: 0 }
        });
        
        annotations.push({
            x: dates[start + foldSize / 2],
            y: 75,
            text: fold === nSplits - 1 ? 'Test' : `Train ${fold + 1}`,
            showarrow: false
        });
    }
    
    const trace = {
        x: dates,
        y: values,
        type: 'scatter',
        mode: 'lines',
        name: 'Patient Vital Signs',
        line: { color: colors.darkPlum, width: 2 }
    };
    
    const layout = {
        title: 'Time Series Split: Respects Temporal Order',
        xaxis: { title: 'Time', color: colors.axisColor, gridcolor: colors.gridColor },
        yaxis: { title: 'Measurement Value', color: colors.axisColor, gridcolor: colors.gridColor },
        shapes: shapes,
        annotations: annotations,
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    // Add description
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Time Series Cross-Validation</h3>
        <p>For temporal medical data: always train on past, test on future.</p>
        <ul>
            <li>✓ Prevents temporal leakage</li>
            <li>✓ Realistic for predictive models</li>
            <li>✓ Suitable for patient monitoring</li>
            <li>⚠️ Less data for early folds</li>
        </ul>
    `;
    container.appendChild(description);
}

/**
 * Method Selector Functionality
 */
function setupMethodSelector() {
    // Initialize with hidden recommendation
    document.getElementById('method-recommendation').style.display = 'none';
}

function selectDataType(type) {
    const recommendationDiv = document.getElementById('method-recommendation');
    const methodsDiv = document.getElementById('recommended-methods');
    
    recommendationDiv.style.display = 'block';
    
    let recommendations = '';
    
    switch(type) {
        case 'iid':
            recommendations = `
                <h3>I.I.D. Data Methods (9 methods)</h3>
                <div class="method-card">
                    <h4>1. Hold-Out (Train-Test Split)</h4>
                    <p>Single split into training and test sets</p>
                    <p>Best for: Quick prototyping, large datasets</p>
                    <code>HoldOut(test_size=0.2)</code>
                </div>
                <div class="method-card">
                    <h4>2. K-Fold Cross-Validation</h4>
                    <p>Standard k-fold splitting</p>
                    <p>Best for: Balanced datasets, general use</p>
                    <code>KFoldMedical(n_splits=5)</code>
                </div>
                <div class="method-card">
                    <h4>3. Stratified K-Fold</h4>
                    <p>Preserves class distribution in each fold</p>
                    <p>Best for: Imbalanced disease classification</p>
                    <code>StratifiedKFoldMedical(n_splits=5)</code>
                </div>
                <div class="method-card">
                    <h4>4. Repeated K-Fold</h4>
                    <p>Multiple k-fold runs with different randomization</p>
                    <p>Best for: Small datasets, variance reduction</p>
                    <code>RepeatedKFold(n_splits=5, n_repeats=10)</code>
                </div>
                <div class="method-card">
                    <h4>5. LOOCV (Leave-One-Out)</h4>
                    <p>Each sample used once as test set</p>
                    <p>Best for: Very small datasets (n < 100)</p>
                    <code>LOOCV()</code>
                </div>
                <div class="method-card">
                    <h4>6. LPOCV (Leave-p-Out)</h4>
                    <p>Leave p samples out for testing</p>
                    <p>Best for: Small datasets with specific test size needs</p>
                    <code>LPOCV(p=2)</code>
                </div>
                <div class="method-card">
                    <h4>7. Bootstrap Validation</h4>
                    <p>Sampling with replacement, includes .632/.632+ estimators</p>
                    <p>Best for: Small datasets, confidence intervals</p>
                    <code>BootstrapValidation(n_iterations=100, estimator='.632')</code>
                </div>
                <div class="method-card">
                    <h4>8. Monte Carlo CV</h4>
                    <p>Random sub-sampling validation</p>
                    <p>Best for: Large datasets, flexible test size</p>
                    <code>MonteCarloCV(n_iterations=100, test_size=0.2)</code>
                </div>
                <div class="method-card">
                    <h4>9. Nested Cross-Validation</h4>
                    <p>Two-level CV for hyperparameter tuning</p>
                    <p>Best for: Model selection with unbiased evaluation</p>
                    <code>NestedCV(outer_cv=KFold(5), inner_cv=KFold(3))</code>
                </div>
            `;
            break;
            
        case 'temporal':
            recommendations = `
                <h3>Temporal Data Methods (8 methods)</h3>
                <div class="method-card">
                    <h4>1. Temporal Clinical Split</h4>
                    <p>Time-aware splitting with optional gap</p>
                    <p>Best for: Disease progression, clinical trials</p>
                    <code>TemporalClinical(n_splits=5, gap=7)</code>
                </div>
                <div class="method-card">
                    <h4>2. Rolling Window CV</h4>
                    <p>Fixed-size window sliding through time</p>
                    <p>Best for: Stable time series, forecasting</p>
                    <code>RollingWindowCV(window_size=100, forecast_horizon=10)</code>
                </div>
                <div class="method-card">
                    <h4>3. Expanding Window CV</h4>
                    <p>Growing training set over time</p>
                    <p>Best for: All historical data relevant</p>
                    <code>ExpandingWindowCV(min_train_size=50)</code>
                </div>
                <div class="method-card">
                    <h4>4. Blocked Time Series</h4>
                    <p>Preserves temporal blocks (days/weeks/months)</p>
                    <p>Best for: Seasonal medical data, clustered events</p>
                    <code>BlockedTimeSeries(n_splits=5, block_size='week')</code>
                </div>
                <div class="method-card">
                    <h4>5. Purged K-Fold CV</h4>
                    <p>K-fold with temporal purging to prevent leakage</p>
                    <p>Best for: Financial-medical data, cost prediction</p>
                    <code>PurgedKFoldCV(n_splits=5, purge_gap=10)</code>
                </div>
                <div class="method-card">
                    <h4>6. Combinatorial Purged CV</h4>
                    <p>Multiple train/test combinations with purging</p>
                    <p>Best for: Robust temporal validation</p>
                    <code>CombinatorialPurgedCV(n_splits=5, n_test_groups=2)</code>
                </div>
                <div class="method-card">
                    <h4>7. Purged Group Time Series</h4>
                    <p>Combines temporal, grouping, and purging</p>
                    <p>Best for: Panel data with time and groups</p>
                    <code>PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=30)</code>
                </div>
                <div class="method-card">
                    <h4>8. Nested Temporal CV</h4>
                    <p>Nested CV respecting temporal order</p>
                    <p>Best for: Time series hyperparameter tuning</p>
                    <code>NestedTemporalCV(outer_cv=ExpandingWindow(), inner_cv=RollingWindow())</code>
                </div>
            `;
            break;
            
        case 'grouped':
            recommendations = `
                <h3>Grouped/Hierarchical Data Methods (8 methods)</h3>
                <div class="method-card">
                    <h4>1. Group K-Fold</h4>
                    <p>All records from a patient stay together</p>
                    <p>Best for: Multiple records per patient</p>
                    <code>GroupKFold(n_splits=5)</code>
                </div>
                <div class="method-card">
                    <h4>2. Stratified Group K-Fold</h4>
                    <p>Preserves class balance with patient grouping</p>
                    <p>Best for: Imbalanced grouped data</p>
                    <code>StratifiedGroupKFold(n_splits=5)</code>
                </div>
                <div class="method-card">
                    <h4>3. Leave-One-Group-Out</h4>
                    <p>Each group (patient/hospital) as test set once</p>
                    <p>Best for: Multi-site trials, new patient generalization</p>
                    <code>LeaveOneGroupOut()</code>
                </div>
                <div class="method-card">
                    <h4>4. Leave-p-Groups-Out</h4>
                    <p>Multiple groups (p) left out for testing</p>
                    <p>Best for: Multi-center validation, robustness testing</p>
                    <code>LeavePGroupsOut(n_groups=2)</code>
                </div>
                <div class="method-card">
                    <h4>5. Repeated Group K-Fold</h4>
                    <p>Multiple runs of group k-fold</p>
                    <p>Best for: Variance reduction in grouped data</p>
                    <code>RepeatedGroupKFold(n_splits=5, n_repeats=10)</code>
                </div>
                <div class="method-card">
                    <h4>6. Hierarchical Group K-Fold</h4>
                    <p>Handles nested structures (Hospital→Department→Patient)</p>
                    <p>Best for: Multi-level medical hierarchies</p>
                    <code>HierarchicalGroupKFold(n_splits=5)</code>
                </div>
                <div class="method-card">
                    <h4>7. Multi-level CV</h4>
                    <p>Cross-validation across multiple hierarchy levels</p>
                    <p>Best for: Site→Department→Patient validation</p>
                    <code>MultilevelCV(n_splits=5, hierarchy_levels=['site', 'department'])</code>
                </div>
                <div class="method-card">
                    <h4>8. Nested Grouped CV</h4>
                    <p>Nested CV preserving group structure</p>
                    <p>Best for: Grouped hyperparameter tuning</p>
                    <code>NestedGroupedCV(outer_cv=GroupKFold(5), inner_cv=GroupKFold(3))</code>
                </div>
            `;
            break;
            
        case 'spatial':
            recommendations = `
                <h3>Spatial/Geographic Data Methods (4 methods)</h3>
                <div class="method-card">
                    <h4>1. Spatial Block CV</h4>
                    <p>Divides space into blocks for validation</p>
                    <p>Best for: Geographic disease spread, environmental health</p>
                    <code>SpatialBlockCV(n_splits=5, block_shape='grid')</code>
                </div>
                <div class="method-card">
                    <h4>2. Buffered Spatial CV</h4>
                    <p>Adds buffer zones to reduce autocorrelation</p>
                    <p>Best for: Strong spatial dependencies</p>
                    <code>BufferedSpatialCV(n_splits=5, buffer_size=2.0)</code>
                </div>
                <div class="method-card">
                    <h4>3. Spatiotemporal Block CV</h4>
                    <p>Combines spatial and temporal blocking</p>
                    <p>Best for: Disease spread over time and space</p>
                    <code>SpatiotemporalBlockCV(n_spatial_blocks=3, n_temporal_blocks=3)</code>
                </div>
                <div class="method-card">
                    <h4>4. Environmental Health CV</h4>
                    <p>Specialized for environmental health studies</p>
                    <p>Best for: Pollution studies, climate health impacts</p>
                    <code>EnvironmentalHealthCV(spatial_blocks=4, temporal_strategy='seasonal')</code>
                </div>
            `;
            break;
    }
    
    methodsDiv.innerHTML = recommendations;
    
    // Smooth scroll to recommendations
    recommendationDiv.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Generate sample patient data
 */
function generatePatientData() {
    const patients = [];
    for (let i = 0; i < 10; i++) {
        patients.push({
            id: `P${i + 1}`,
            records: Math.floor(Math.random() * 5) + 2,
            class: Math.random() > 0.3 ? 0 : 1
        });
    }
    return patients;
}

/**
 * Copy code functionality
 */
function copyCode() {
    const codeBlock = document.querySelector('.code-example code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        // Show feedback
        const btn = event.target.closest('button');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="material-icons">check</span> Copied!';
        
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}

/**
 * Start interactive tutorial
 */
function startInteractiveTutorial() {
    // Scroll to method selector
    document.querySelector('.method-selector').scrollIntoView({ 
        behavior: 'smooth' 
    });
    
    // Highlight the section
    const section = document.querySelector('.method-selector');
    section.style.animation = 'pulse 2s';
    
    setTimeout(() => {
        section.style.animation = '';
    }, 2000);
}

/**
 * Update method options based on data type selection
 */
function updateMethodOptions() {
    const dataType = document.getElementById('data-type').value;
    const methodSelect = document.getElementById('cv-method');
    
    let options = '';
    
    switch(dataType) {
        case 'iid':
            options = `
                <option value="holdout">Hold-Out (Train-Test Split)</option>
                <option value="kfold" selected>K-Fold CV</option>
                <option value="stratified">Stratified K-Fold</option>
                <option value="repeated">Repeated K-Fold</option>
                <option value="loocv">LOOCV (Leave-One-Out)</option>
                <option value="lpocv">LPOCV (Leave-p-Out)</option>
                <option value="bootstrap">Bootstrap Validation</option>
                <option value="montecarlo">Monte Carlo CV</option>
                <option value="nested">Nested CV</option>
            `;
            break;
        case 'temporal':
            options = `
                <option value="temporal" selected>Temporal Clinical</option>
                <option value="rolling">Rolling Window</option>
                <option value="expanding">Expanding Window</option>
                <option value="blocked">Blocked Time Series</option>
                <option value="purged">Purged K-Fold</option>
                <option value="cpcv">Combinatorial Purged</option>
                <option value="purged_group">Purged Group Time Series</option>
                <option value="nested_temporal">Nested Temporal CV</option>
            `;
            break;
        case 'grouped':
            options = `
                <option value="grouped" selected>Patient Group K-Fold</option>
                <option value="stratified_grouped">Stratified Group K-Fold</option>
                <option value="logo">Leave-One-Group-Out</option>
                <option value="lpgo">Leave-p-Groups-Out</option>
                <option value="repeated_grouped">Repeated Group K-Fold</option>
                <option value="hierarchical">Hierarchical Group K-Fold</option>
                <option value="multilevel">Multi-level CV</option>
                <option value="nested_grouped">Nested Grouped CV</option>
            `;
            break;
        case 'spatial':
            options = `
                <option value="spatial_block" selected>Spatial Block CV</option>
                <option value="buffered_spatial">Buffered Spatial CV</option>
                <option value="spatiotemporal">Spatiotemporal Block CV</option>
                <option value="environmental">Environmental Health CV</option>
            `;
            break;
    }
    
    methodSelect.innerHTML = options;
    updateVisualization();
}

// Initialize method options on page load
document.addEventListener('DOMContentLoaded', function() {
    updateMethodOptions();
});

/**
 * Additional visualization functions for all CV methods
 */

// I.I.D. Methods
function visualizeHoldOut() {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Create simple train/test split visualization
    const data = [
        {x: ['Training Set (80%)'], y: [80], type: 'bar', marker: {color: colors.plum}},
        {x: ['Test Set (20%)'], y: [20], type: 'bar', marker: {color: colors.orange}}
    ];
    
    const layout = {
        title: 'Hold-Out Validation (Train-Test Split)',
        barmode: 'stack',
        yaxis: {title: 'Percentage of Data', color: colors.axisColor, gridcolor: colors.gridColor},
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Hold-Out Validation</h3>
        <p>Simple single split of data into training and test sets.</p>
        <ul>
            <li>✓ Fast and simple</li>
            <li>✓ Good for large datasets</li>
            <li>✗ High variance in performance estimate</li>
            <li>✗ Not all data used for training</li>
        </ul>
    `;
    container.appendChild(description);
}

function visualizeRepeatedKFold(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Show multiple repetitions
    const repeats = 3;
    const traces = [];
    
    for (let r = 0; r < repeats; r++) {
        traces.push({
            x: Array.from({length: nSplits}, (_, i) => `Fold ${i+1}`),
            y: Array.from({length: nSplits}, () => 85 + Math.random() * 10),
            name: `Repeat ${r+1}`,
            type: 'scatter',
            mode: 'lines+markers'
        });
    }
    
    const layout = {
        title: 'Repeated K-Fold: Multiple Runs with Different Randomization',
        xaxis: {title: 'Fold', color: colors.axisColor, gridcolor: colors.gridColor},
        yaxis: {title: 'Accuracy (%)', color: colors.axisColor, gridcolor: colors.gridColor},
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Repeated K-Fold Cross-Validation</h3>
        <p>Performs k-fold CV multiple times with different random splits.</p>
        <ul>
            <li>✓ Reduces variance in estimates</li>
            <li>✓ More robust evaluation</li>
            <li>✗ Computationally expensive</li>
        </ul>
    `;
    container.appendChild(description);
}

function visualizeLOOCV() {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Show leave-one-out concept
    const n = 10; // Small dataset for visualization
    const foldDiv = document.createElement('div');
    foldDiv.className = 'fold-visualization';
    
    for (let i = 0; i < Math.min(5, n); i++) { // Show first 5 iterations
        const foldContainer = document.createElement('div');
        foldContainer.className = 'fold-container';
        foldContainer.innerHTML = `<h4>Iteration ${i + 1}</h4>`;
        
        const samplesDiv = document.createElement('div');
        samplesDiv.className = 'samples-container';
        
        for (let j = 0; j < n; j++) {
            const sample = document.createElement('div');
            sample.className = 'sample';
            sample.style.backgroundColor = j === i ? colors.orange : colors.plum;
            sample.title = j === i ? 'Test' : 'Train';
            samplesDiv.appendChild(sample);
        }
        
        foldContainer.appendChild(samplesDiv);
        foldDiv.appendChild(foldContainer);
    }
    
    container.appendChild(foldDiv);
    
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Leave-One-Out Cross-Validation</h3>
        <p>Each sample is used once as the test set. N iterations for N samples.</p>
        <ul>
            <li>✓ Maximum data usage</li>
            <li>✓ Nearly unbiased estimate</li>
            <li>✗ Very expensive (O(n²))</li>
            <li>✗ High variance</li>
        </ul>
    `;
    container.appendChild(description);
}

function visualizeLPOCV() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Leave-p-Out Cross-Validation</h3>
            <p>Generalization of LOOCV where p samples are left out each iteration.</p>
            <p>Number of iterations: C(n,p) = n!/(p!(n-p)!)</p>
            <ul>
                <li>✓ Flexible test set size</li>
                <li>✓ Thorough evaluation</li>
                <li>✗ Combinatorial explosion</li>
                <li>✗ Extremely expensive for large p</li>
            </ul>
        </div>
    `;
}

function visualizeBootstrap() {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Show bootstrap sampling concept
    const data = [{
        labels: ['In-Bag', 'Out-of-Bag'],
        values: [63.2, 36.8],
        type: 'pie',
        marker: {
            colors: [colors.plum, colors.orange]
        }
    }];
    
    const layout = {
        title: 'Bootstrap Sampling: ~63.2% In-Bag, ~36.8% OOB',
        paper_bgcolor: colors.paperBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Bootstrap Validation</h3>
        <p>Sampling with replacement. Includes .632 and .632+ estimators.</p>
        <ul>
            <li>✓ Good for small datasets</li>
            <li>✓ Provides confidence intervals</li>
            <li>✓ .632+ handles overfitting</li>
            <li>✗ Biased for small samples</li>
        </ul>
    `;
    container.appendChild(description);
}

function visualizeMonteCarlo() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Monte Carlo Cross-Validation</h3>
            <p>Random sub-sampling validation with flexible test size.</p>
            <ul>
                <li>✓ Flexible train/test ratio</li>
                <li>✓ Can run many iterations</li>
                <li>✓ Good for large datasets</li>
                <li>✗ Test sets may overlap</li>
            </ul>
        </div>
    `;
}

function visualizeNested() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Nested Cross-Validation</h3>
            <p>Two-level CV: outer loop for evaluation, inner loop for hyperparameter tuning.</p>
            <ul>
                <li>✓ Unbiased performance estimate</li>
                <li>✓ Proper model selection</li>
                <li>✗ Computationally expensive</li>
                <li>✗ Complex to implement</li>
            </ul>
        </div>
    `;
}

// Temporal Methods
function visualizeRollingWindow() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Rolling Window Cross-Validation</h3>
            <p>Fixed-size training window slides through time.</p>
            <ul>
                <li>✓ Constant training size</li>
                <li>✓ Good for stationary series</li>
                <li>✗ Discards old data</li>
            </ul>
        </div>
    `;
}

function visualizeExpandingWindow() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Expanding Window Cross-Validation</h3>
            <p>Training set grows over time, always starting from beginning.</p>
            <ul>
                <li>✓ Uses all historical data</li>
                <li>✓ Good for learning curves</li>
                <li>✗ Varying training size</li>
            </ul>
        </div>
    `;
}

function visualizeBlockedTimeSeries(nSplits) {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Blocked Time Series Cross-Validation</h3>
            <p>Preserves temporal blocks (days, weeks, months) together.</p>
            <ul>
                <li>✓ Handles seasonal patterns</li>
                <li>✓ Preserves temporal structure</li>
                <li>✗ May have uneven splits</li>
            </ul>
        </div>
    `;
}

function visualizePurgedKFold(nSplits) {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Purged K-Fold Cross-Validation</h3>
            <p>K-fold with temporal gap (purge) between train and test to prevent leakage.</p>
            <ul>
                <li>✓ Prevents information leakage</li>
                <li>✓ Good for financial data</li>
                <li>✗ Reduces training data</li>
            </ul>
        </div>
    `;
}

function visualizeCPCV() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Combinatorial Purged Cross-Validation</h3>
            <p>Multiple train/test combinations with purging.</p>
            <ul>
                <li>✓ Robust validation</li>
                <li>✓ Many test scenarios</li>
                <li>✗ Computationally intensive</li>
            </ul>
        </div>
    `;
}

function visualizePurgedGroupTimeSeries() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Purged Group Time Series Split</h3>
            <p>Combines temporal order, patient grouping, and purging.</p>
            <ul>
                <li>✓ Handles panel data</li>
                <li>✓ Prevents all leakage types</li>
                <li>✗ Complex to implement</li>
            </ul>
        </div>
    `;
}

function visualizeNestedTemporal() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Nested Temporal Cross-Validation</h3>
            <p>Nested CV preserving temporal order in both loops.</p>
            <ul>
                <li>✓ Proper time series tuning</li>
                <li>✓ No future information leak</li>
                <li>✗ Very expensive</li>
            </ul>
        </div>
    `;
}

// Grouped Methods
function visualizeStratifiedGrouped(nSplits) {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Stratified Group K-Fold</h3>
            <p>Maintains class balance while keeping patient groups together.</p>
            <ul>
                <li>✓ Preserves class distribution</li>
                <li>✓ No patient leakage</li>
                <li>✗ Complex balancing</li>
            </ul>
        </div>
    `;
}

function visualizeLOGO() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Leave-One-Group-Out CV</h3>
            <p>Each group (patient/hospital) used once as test set.</p>
            <ul>
                <li>✓ Tests generalization to new groups</li>
                <li>✓ No group contamination</li>
                <li>✗ Many iterations</li>
            </ul>
        </div>
    `;
}

function visualizeLPGO() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Leave-p-Groups-Out CV</h3>
            <p>Leave p groups out for testing at each iteration.</p>
            <ul>
                <li>✓ Tests on multiple groups at once</li>
                <li>✓ Good for multi-center studies</li>
                <li>✗ Many iterations: C(n,p)</li>
                <li>✗ Expensive for large number of groups</li>
            </ul>
        </div>
    `;
}

function visualizeRepeatedGrouped(nSplits) {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Repeated Group K-Fold</h3>
            <p>Multiple runs of group k-fold with different randomization.</p>
            <ul>
                <li>✓ Reduces variance</li>
                <li>✓ Maintains grouping</li>
                <li>✗ Computationally expensive</li>
            </ul>
        </div>
    `;
}

function visualizeHierarchical() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Hierarchical Group K-Fold</h3>
            <p>Handles nested structures: Hospital → Department → Patient.</p>
            <ul>
                <li>✓ Multi-level validation</li>
                <li>✓ Respects hierarchy</li>
                <li>✗ Complex implementation</li>
            </ul>
        </div>
    `;
}

function visualizeMultilevel() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Multi-level Cross-Validation</h3>
            <p>Cross-validation across multiple hierarchy levels in medical data.</p>
            <ul>
                <li>✓ Site → Department → Patient validation</li>
                <li>✓ Tests generalization at each level</li>
                <li>✓ Proper nested structure handling</li>
                <li>✗ Requires clear hierarchy definition</li>
            </ul>
        </div>
    `;
}

function visualizeNestedGrouped() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Nested Grouped Cross-Validation</h3>
            <p>Nested CV maintaining group structure in both loops.</p>
            <ul>
                <li>✓ Proper grouped tuning</li>
                <li>✓ No group leakage</li>
                <li>✗ Very expensive</li>
            </ul>
        </div>
    `;
}

// Spatial Methods
function visualizeSpatialBlock(nSplits) {
    var colors = getColors();
    const container = document.getElementById('cv-visualization');

    // Create spatial grid visualization
    const gridSize = Math.ceil(Math.sqrt(nSplits));
    const data = [];
    
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const blockId = i * gridSize + j;
            if (blockId < nSplits) {
                data.push({
                    x: [j, j+1, j+1, j, j],
                    y: [i, i, i+1, i+1, i],
                    fill: 'toself',
                    fillcolor: blockId === 0 ? colors.orange : colors.plum,
                    line: {color: 'white', width: 2},
                    name: `Block ${blockId + 1}`,
                    showlegend: blockId < 2
                });
            }
        }
    }
    
    const layout = {
        title: 'Spatial Block Cross-Validation',
        xaxis: {title: 'Longitude', showgrid: false, color: colors.axisColor},
        yaxis: {title: 'Latitude', showgrid: false, color: colors.axisColor},
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.text }
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    const description = document.createElement('div');
    description.className = 'method-description';
    description.innerHTML = `
        <h3>Spatial Block Cross-Validation</h3>
        <p>Divides geographic space into blocks for validation.</p>
        <ul>
            <li>✓ Handles spatial autocorrelation</li>
            <li>✓ Tests spatial generalization</li>
            <li>✗ May have uneven sample distribution</li>
        </ul>
    `;
    container.appendChild(description);
}

function visualizeBufferedSpatial(nSplits) {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Buffered Spatial Cross-Validation</h3>
            <p>Adds buffer zones around test blocks to reduce autocorrelation.</p>
            <ul>
                <li>✓ Stronger independence</li>
                <li>✓ Better generalization test</li>
                <li>✗ Loses training data in buffers</li>
            </ul>
        </div>
    `;
}

function visualizeSpatiotemporal() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Spatiotemporal Block Cross-Validation</h3>
            <p>Combines spatial and temporal blocking for 4D data.</p>
            <ul>
                <li>✓ Handles space-time dependencies</li>
                <li>✓ Good for disease spread</li>
                <li>✗ Complex block structure</li>
            </ul>
        </div>
    `;
}

function visualizeEnvironmental() {
    const container = document.getElementById('cv-visualization');
    container.innerHTML = `
        <div class="method-description">
            <h3>Environmental Health Cross-Validation</h3>
            <p>Specialized for environmental health studies with multiple factors.</p>
            <ul>
                <li>✓ Handles environmental covariates</li>
                <li>✓ Seasonal considerations</li>
                <li>✗ Domain-specific</li>
            </ul>
        </div>
    `;
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(135, 0, 82, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(135, 0, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(135, 0, 82, 0); }
    }
    
    .fold-visualization {
        padding: 20px;
    }
    
    .fold-container {
        margin-bottom: 15px;
    }
    
    .fold-container h4 {
        color: var(--ki-plum);
        margin-bottom: 10px;
    }
    
    .samples-container {
        display: flex;
        gap: 2px;
        flex-wrap: wrap;
    }
    
    .sample {
        width: 15px;
        height: 15px;
        border-radius: 3px;
        transition: transform 0.2s;
    }
    
    .sample:hover {
        transform: scale(1.5);
    }
    
    .method-description {
        margin-top: 30px;
        padding: 20px;
        background: var(--card-bg);
        border-radius: 12px;
        box-shadow: var(--md-sys-elevation-1);
        border: 1px solid var(--border-color);
    }
    
    .method-description h3 {
        color: var(--ki-plum);
        margin-bottom: 10px;
    }
    
    .method-description ul {
        margin-top: 15px;
        padding-left: 20px;
    }
    
    .method-description li {
        margin-bottom: 8px;
        color: var(--ki-grey);
    }
    
    .method-card {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: var(--md-sys-elevation-1);
        border: 1px solid var(--border-color);
    }
    
    .method-card h4 {
        color: var(--ki-plum);
        margin-bottom: 8px;
    }
    
    .method-card code {
        background: var(--ki-light-blue);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        color: var(--ki-dark-plum);
    }
`;
document.head.appendChild(style);

// Re-render visualizations when theme changes
window.addEventListener('themechange', function() {
    if (typeof updateVisualization === 'function') {
        try { updateVisualization(); } catch(e) {}
    }
    if (typeof initHeroVisualization === 'function') {
        try { initHeroVisualization(); } catch(e) {}
    }
});