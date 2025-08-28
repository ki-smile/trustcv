/**
 * Cross-Validation Visualization Functions
 * All visualization methods for different CV strategies
 */

// KI Color Palette (imported from main)
const colors = {
    darkPlum: '#4F0433',
    plum: '#870052',
    orange: '#FF876F',
    lightOrange: '#FEEEEB',
    lightBlue: '#EDF4F4',
    grey: '#6B6B6B'
};

/**
 * I.I.D. Methods Visualizations
 */

function visualizeHoldOut() {
    const container = document.getElementById('cv-visualization');
    
    const data = [
        {x: ['Training Set (80%)'], y: [80], type: 'bar', marker: {color: colors.plum}},
        {x: ['Test Set (20%)'], y: [20], type: 'bar', marker: {color: colors.orange}}
    ];
    
    const layout = {
        title: 'Hold-Out Validation (Train-Test Split)',
        barmode: 'stack',
        yaxis: {title: 'Percentage of Data'},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    addMethodDescription(container, 'Hold-Out Validation', 
        'Simple single split of data into training and test sets.',
        ['Fast and simple', 'Good for large datasets'],
        ['High variance in performance estimate', 'Not all data used for training']
    );
}

function visualizeKFold(nSplits) {
    const container = document.getElementById('cv-visualization');
    const nSamples = 50;
    
    const foldDiv = document.createElement('div');
    foldDiv.className = 'fold-visualization';
    
    for (let fold = 0; fold < nSplits; fold++) {
        const foldContainer = createFoldContainer(fold + 1, nSamples, nSplits, fold);
        foldDiv.appendChild(foldContainer);
    }
    
    container.appendChild(foldDiv);
    addMethodDescription(container, 'K-Fold Cross-Validation',
        `Data is divided into ${nSplits} equal folds. Each fold serves as test set once.`,
        ['All data used for training and testing', 'Reduced variance in performance estimate'],
        ['May not preserve class distribution', 'Not suitable for grouped or temporal data']
    );
}

function visualizeStratified(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    const data = [];
    for (let fold = 0; fold < nSplits; fold++) {
        data.push({
            x: ['Class 0', 'Class 1'],
            y: [30, 20],
            name: `Fold ${fold + 1}`,
            type: 'bar',
            marker: {color: fold === 0 ? colors.orange : colors.plum}
        });
    }
    
    const layout = {
        title: 'Stratified K-Fold: Class Distribution Preserved',
        barmode: 'group',
        xaxis: {title: 'Class'},
        yaxis: {title: 'Number of Samples'},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    addMethodDescription(container, 'Stratified K-Fold Cross-Validation',
        'Preserves the class distribution in each fold - critical for imbalanced medical datasets.',
        ['Maintains class proportions', 'Better for imbalanced datasets', 'More reliable performance estimates'],
        ['Still not suitable for grouped data']
    );
}

function visualizeRepeatedKFold(nSplits) {
    const container = document.getElementById('cv-visualization');
    
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
        xaxis: {title: 'Fold'},
        yaxis: {title: 'Accuracy (%)'},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    addMethodDescription(container, 'Repeated K-Fold Cross-Validation',
        'Performs k-fold CV multiple times with different random splits.',
        ['Reduces variance in estimates', 'More robust evaluation'],
        ['Computationally expensive']
    );
}

function visualizeLOOCV() {
    const container = document.getElementById('cv-visualization');
    
    const n = 10;
    const foldDiv = document.createElement('div');
    foldDiv.className = 'fold-visualization';
    
    for (let i = 0; i < Math.min(5, n); i++) {
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
    addMethodDescription(container, 'Leave-One-Out Cross-Validation',
        'Each sample is used once as the test set. N iterations for N samples.',
        ['Maximum data usage', 'Nearly unbiased estimate'],
        ['Very expensive (O(n²))', 'High variance']
    );
}

function visualizeLPOCV() {
    showTextDescription('Leave-p-Out Cross-Validation',
        'Generalization of LOOCV where p samples are left out each iteration.',
        'Number of iterations: C(n,p) = n!/(p!(n-p)!)',
        ['Flexible test set size', 'Thorough evaluation'],
        ['Combinatorial explosion', 'Extremely expensive for large p']
    );
}

function visualizeBootstrap() {
    const container = document.getElementById('cv-visualization');
    
    const data = [{
        labels: ['In-Bag', 'Out-of-Bag'],
        values: [63.2, 36.8],
        type: 'pie',
        marker: {colors: [colors.plum, colors.orange]}
    }];
    
    const layout = {
        title: 'Bootstrap Sampling: ~63.2% In-Bag, ~36.8% OOB',
        paper_bgcolor: colors.lightBlue
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    addMethodDescription(container, 'Bootstrap Validation',
        'Sampling with replacement. Includes .632 and .632+ estimators.',
        ['Good for small datasets', 'Provides confidence intervals', '.632+ handles overfitting'],
        ['Biased for small samples']
    );
}

function visualizeMonteCarlo() {
    showTextDescription('Monte Carlo Cross-Validation',
        'Random sub-sampling validation with flexible test size.',
        '',
        ['Flexible train/test ratio', 'Can run many iterations', 'Good for large datasets'],
        ['Test sets may overlap']
    );
}

function visualizeNested() {
    showTextDescription('Nested Cross-Validation',
        'Two-level CV: outer loop for evaluation, inner loop for hyperparameter tuning.',
        '',
        ['Unbiased performance estimate', 'Proper model selection'],
        ['Computationally expensive', 'Complex to implement']
    );
}

/**
 * Helper Functions
 */

function createFoldContainer(foldNum, nSamples, nSplits, currentFold) {
    const foldContainer = document.createElement('div');
    foldContainer.className = 'fold-container';
    foldContainer.innerHTML = `<h4>Fold ${foldNum}</h4>`;
    
    const samplesDiv = document.createElement('div');
    samplesDiv.className = 'samples-container';
    
    const foldSize = Math.floor(nSamples / nSplits);
    const testStart = currentFold * foldSize;
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
    return foldContainer;
}

function addMethodDescription(container, title, description, pros, cons) {
    const descDiv = document.createElement('div');
    descDiv.className = 'method-description';
    
    let html = `<h3>${title}</h3><p>${description}</p><ul>`;
    pros.forEach(pro => html += `<li>✓ ${pro}</li>`);
    cons.forEach(con => html += `<li>✗ ${con}</li>`);
    html += '</ul>';
    
    descDiv.innerHTML = html;
    container.appendChild(descDiv);
}

function showTextDescription(title, description, extra, pros, cons) {
    const container = document.getElementById('cv-visualization');
    let html = `<div class="method-description">
        <h3>${title}</h3>
        <p>${description}</p>`;
    
    if (extra) html += `<p>${extra}</p>`;
    
    html += '<ul>';
    pros.forEach(pro => html += `<li>✓ ${pro}</li>`);
    cons.forEach(con => html += `<li>✗ ${con}</li>`);
    html += '</ul></div>';
    
    container.innerHTML = html;
}