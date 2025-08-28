/**
 * Grouped/Hierarchical Cross-Validation Visualizations
 */

function visualizeGrouped(nSplits) {
    const container = document.getElementById('cv-visualization');
    
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
            marker: {color: colors_palette[fold]}
        });
    }
    
    const layout = {
        title: 'Patient-Grouped K-Fold: No Patient Appears in Multiple Folds',
        barmode: 'stack',
        xaxis: {title: 'Patient ID'},
        yaxis: {title: 'Number of Records'},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    addMethodDescription(container, 'Patient-Grouped K-Fold Cross-Validation',
        'Essential for medical data: ensures all records from a patient stay together.',
        ['Prevents patient data leakage', 'More realistic performance estimate', 'Required for longitudinal studies'],
        ['May result in uneven fold sizes']
    );
}

function visualizeStratifiedGrouped(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    // Show class balance preservation with grouping
    const data = [];
    const groups = ['Healthy Groups', 'Disease Groups'];
    
    for (let fold = 0; fold < Math.min(nSplits, 3); fold++) {
        data.push({
            x: groups,
            y: [60, 40], // Preserved ratio
            name: `Fold ${fold + 1}`,
            type: 'bar',
            marker: {color: fold === 0 ? colors.orange : colors.plum}
        });
    }
    
    const layout = {
        title: 'Stratified Group K-Fold: Preserves Class Balance & Groups',
        barmode: 'group',
        xaxis: {title: 'Patient Groups'},
        yaxis: {title: 'Percentage'},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    addMethodDescription(container, 'Stratified Group K-Fold',
        'Maintains class balance while keeping patient groups together.',
        ['Preserves class distribution', 'No patient leakage'],
        ['Complex balancing']
    );
}

function visualizeLOGO() {
    const container = document.getElementById('cv-visualization');
    
    // Show leave-one-group-out concept
    const nGroups = 6;
    const groupNames = Array.from({length: nGroups}, (_, i) => `Hospital ${i + 1}`);
    
    const data = [];
    
    for (let i = 0; i < Math.min(3, nGroups); i++) {
        const y = Array(nGroups).fill(1);
        y[i] = 0; // This group is test
        
        data.push({
            x: groupNames,
            y: y,
            name: `Iteration ${i + 1}`,
            type: 'bar',
            marker: {
                color: y.map(v => v === 0 ? colors.orange : colors.plum)
            },
            showlegend: i === 0
        });
    }
    
    const layout = {
        title: 'Leave-One-Group-Out: Each Hospital as Test Set Once',
        xaxis: {title: 'Hospital/Group'},
        yaxis: {title: 'Role', ticktext: ['Test', 'Train'], tickvals: [0, 1]},
        barmode: 'group',
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    addMethodDescription(container, 'Leave-One-Group-Out CV',
        'Each group (patient/hospital) used once as test set.',
        ['Tests generalization to new groups', 'No group contamination'],
        ['Many iterations']
    );
}

function visualizeRepeatedGrouped(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    // Show multiple repetitions of group k-fold
    const nRepeats = 3;
    const nGroups = 10;
    const traces = [];
    
    // Generate data for each repeat
    for (let repeat = 0; repeat < nRepeats; repeat++) {
        const accuracies = [];
        const labels = [];
        
        for (let fold = 0; fold < nSplits; fold++) {
            labels.push(`R${repeat + 1}-F${fold + 1}`);
            // Simulate different accuracies for each repeat/fold
            accuracies.push(75 + Math.random() * 15 + repeat * 2);
        }
        
        traces.push({
            x: labels,
            y: accuracies,
            name: `Repeat ${repeat + 1}`,
            type: 'scatter',
            mode: 'lines+markers',
            line: {width: 2},
            marker: {size: 8}
        });
    }
    
    const layout = {
        title: 'Repeated Group K-Fold: Multiple Runs with Patient Grouping',
        xaxis: {title: 'Repeat-Fold'},
        yaxis: {title: 'Accuracy (%)', range: [70, 95]},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white',
        hovermode: 'x unified'
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    // Add visual representation of group preservation
    const groupDiv = document.createElement('div');
    groupDiv.style.marginTop = '20px';
    groupDiv.innerHTML = '<h4>Group Preservation Across Repeats</h4>';
    
    const gridDiv = document.createElement('div');
    gridDiv.style.display = 'grid';
    gridDiv.style.gridTemplateColumns = `repeat(${nRepeats}, 1fr)`;
    gridDiv.style.gap = '10px';
    gridDiv.style.marginTop = '10px';
    
    // Show how groups are shuffled in each repeat
    for (let repeat = 0; repeat < nRepeats; repeat++) {
        const repeatDiv = document.createElement('div');
        repeatDiv.style.border = '1px solid ' + colors.grey;
        repeatDiv.style.borderRadius = '8px';
        repeatDiv.style.padding = '10px';
        repeatDiv.style.backgroundColor = 'white';
        
        repeatDiv.innerHTML = `<h5>Repeat ${repeat + 1}</h5>`;
        
        const foldsContainer = document.createElement('div');
        
        // Simulate random group assignment for this repeat
        const groups = Array.from({length: nGroups}, (_, i) => i);
        // Shuffle groups differently for each repeat
        const shuffled = [...groups].sort(() => Math.random() - 0.5);
        
        for (let fold = 0; fold < Math.min(nSplits, 3); fold++) {
            const foldDiv = document.createElement('div');
            foldDiv.style.marginTop = '5px';
            foldDiv.style.fontSize = '12px';
            
            const foldGroups = [];
            for (let g = fold; g < nGroups; g += nSplits) {
                foldGroups.push(`P${shuffled[g] + 1}`);
            }
            
            foldDiv.innerHTML = `
                <span style="color: ${colors.plum}; font-weight: bold;">Fold ${fold + 1}:</span>
                <span style="color: ${colors.grey};">${foldGroups.slice(0, 3).join(', ')}${foldGroups.length > 3 ? '...' : ''}</span>
            `;
            foldsContainer.appendChild(foldDiv);
        }
        
        repeatDiv.appendChild(foldsContainer);
        gridDiv.appendChild(repeatDiv);
    }
    
    groupDiv.appendChild(gridDiv);
    container.appendChild(groupDiv);
    
    addMethodDescription(container, 'Repeated Group K-Fold Cross-Validation',
        'Multiple runs of group k-fold with different random patient group assignments each time.',
        ['Reduces variance in estimates', 'Maintains patient grouping integrity', 'More robust evaluation'],
        ['Computationally expensive', 'Requires sufficient number of groups']
    );
}

function visualizeHierarchical() {
    const container = document.getElementById('cv-visualization');
    
    // Show hierarchical structure
    const data = [{
        type: 'sunburst',
        labels: ['Medical Center', 'Hospital A', 'Hospital B', 'Dept A1', 'Dept A2', 'Dept B1', 
                'Patient 1', 'Patient 2', 'Patient 3', 'Patient 4', 'Patient 5', 'Patient 6'],
        parents: ['', 'Medical Center', 'Medical Center', 'Hospital A', 'Hospital A', 'Hospital B',
                 'Dept A1', 'Dept A1', 'Dept A2', 'Dept B1', 'Dept B1', 'Dept B1'],
        values: [0, 0, 0, 0, 0, 0, 10, 15, 8, 12, 14, 11],
        marker: {
            colors: [colors.darkPlum, colors.plum, colors.plum, colors.orange, colors.orange, colors.orange,
                    colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange]
        }
    }];
    
    const layout = {
        title: 'Hierarchical Structure: Hospital → Department → Patient',
        paper_bgcolor: colors.lightBlue
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    addMethodDescription(container, 'Hierarchical Group K-Fold',
        'Handles nested structures: Hospital → Department → Patient.',
        ['Multi-level validation', 'Respects hierarchy'],
        ['Complex implementation']
    );
}

function visualizeNestedGrouped() {
    showTextDescription('Nested Grouped Cross-Validation',
        'Nested CV maintaining group structure in both loops.',
        '',
        ['Proper grouped tuning', 'No group leakage'],
        ['Very expensive']
    );
}

function visualizeLPGO(p = 2) {
    const container = document.getElementById('cv-visualization');
    const nGroups = 10;
    const groupNames = Array.from({length: nGroups}, (_, i) => `Group ${i + 1}`);
    
    // Show multiple iterations of Leave-p-Groups-Out
    const traces = [];
    const iterations = Math.min(6, Math.floor(nGroups * (nGroups - 1) / (p * (p - 1))));
    
    for (let iter = 0; iter < iterations; iter++) {
        const testGroups = new Set();
        // Select p groups for testing
        while (testGroups.size < p) {
            testGroups.add(Math.floor(Math.random() * nGroups));
        }
        
        const y = Array(nGroups).fill(0).map((_, i) => testGroups.has(i) ? 0 : 1);
        
        traces.push({
            x: groupNames,
            y: y,
            name: `Iteration ${iter + 1}`,
            type: 'bar',
            marker: {
                color: y.map(v => v === 0 ? colors.orange : colors.plum)
            },
            showlegend: iter < 3
        });
    }
    
    const layout = {
        title: `Leave-${p}-Groups-Out: ${p} Groups as Test Set Each Iteration`,
        xaxis: {title: 'Group'},
        yaxis: {title: 'Role', ticktext: ['Test', 'Train'], tickvals: [0, 1]},
        barmode: 'group',
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white',
        annotations: [{
            text: `Total iterations: C(${nGroups},${p}) = ${Math.floor(factorial(nGroups) / (factorial(p) * factorial(nGroups - p)))}`,
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: -0.15,
            showarrow: false
        }]
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    addMethodDescription(container, 'Leave-p-Groups-Out CV',
        `Each combination of ${p} groups used as test set. Exhaustive validation across group combinations.`,
        ['Tests all group combinations', 'Thorough validation', 'Good for small group counts'],
        [`Many iterations (C(n,${p}))`, 'Computationally expensive', 'Not feasible for large n']
    );
}

function visualizeMultilevel() {
    const container = document.getElementById('cv-visualization');
    
    // Create hierarchical tree visualization
    const data = [{
        type: 'treemap',
        labels: [
            'Medical System',
            'Hospital A', 'Hospital B', 'Hospital C',
            'Dept A1', 'Dept A2', 'Dept B1', 'Dept B2', 'Dept C1',
            'Team A1-1', 'Team A1-2', 'Team A2-1', 'Team B1-1', 'Team B2-1', 'Team C1-1', 'Team C1-2',
            'Pat 1', 'Pat 2', 'Pat 3', 'Pat 4', 'Pat 5', 'Pat 6', 'Pat 7', 'Pat 8'
        ],
        parents: [
            '',
            'Medical System', 'Medical System', 'Medical System',
            'Hospital A', 'Hospital A', 'Hospital B', 'Hospital B', 'Hospital C',
            'Dept A1', 'Dept A1', 'Dept A2', 'Dept B1', 'Dept B2', 'Dept C1', 'Dept C1',
            'Team A1-1', 'Team A1-1', 'Team A1-2', 'Team A2-1', 'Team B1-1', 'Team B2-1', 'Team C1-1', 'Team C1-2'
        ],
        values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 12, 8, 15, 11, 14, 9, 13],
        text: [
            'System Level', 
            'Site Level', 'Site Level', 'Site Level',
            'Department', 'Department', 'Department', 'Department', 'Department',
            'Team', 'Team', 'Team', 'Team', 'Team', 'Team', 'Team',
            'Patient', 'Patient', 'Patient', 'Patient', 'Patient', 'Patient', 'Patient', 'Patient'
        ],
        textposition: 'middle center',
        marker: {
            colors: [
                colors.darkPlum,
                colors.plum, colors.plum, colors.plum,
                colors.orange, colors.orange, colors.orange, colors.orange, colors.orange,
                colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange, colors.lightOrange,
                '#FFE5E0', '#FFE5E0', '#FFE5E0', '#FFE5E0', '#FFE5E0', '#FFE5E0', '#FFE5E0', '#FFE5E0'
            ]
        },
        hovertemplate: '<b>%{label}</b><br>%{text}<br>Records: %{value}<extra></extra>'
    }];
    
    const layout = {
        title: 'Multi-level Hierarchy: System → Hospital → Department → Team → Patient',
        paper_bgcolor: colors.lightBlue,
        margin: {t: 50, l: 0, r: 0, b: 0}
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    // Add level selection visualization
    const levelDiv = document.createElement('div');
    levelDiv.style.marginTop = '20px';
    levelDiv.innerHTML = '<h4>Validation at Different Hierarchy Levels</h4>';
    
    const levels = [
        {name: 'Patient Level', desc: 'Standard CV, may leak team/dept info', color: '#FFE5E0'},
        {name: 'Team Level', desc: 'Groups by team, preserves team integrity', color: colors.lightOrange},
        {name: 'Department Level', desc: 'Groups by department, tests dept generalization', color: colors.orange},
        {name: 'Hospital Level', desc: 'Groups by hospital, tests site generalization', color: colors.plum}
    ];
    
    const gridDiv = document.createElement('div');
    gridDiv.style.display = 'grid';
    gridDiv.style.gridTemplateColumns = 'repeat(2, 1fr)';
    gridDiv.style.gap = '10px';
    gridDiv.style.marginTop = '10px';
    
    levels.forEach(level => {
        const levelCard = document.createElement('div');
        levelCard.style.border = `2px solid ${level.color}`;
        levelCard.style.borderRadius = '8px';
        levelCard.style.padding = '10px';
        levelCard.style.backgroundColor = 'white';
        
        levelCard.innerHTML = `
            <h5 style="color: ${colors.darkPlum}; margin: 0;">${level.name}</h5>
            <p style="font-size: 12px; color: ${colors.grey}; margin: 5px 0 0 0;">${level.desc}</p>
        `;
        gridDiv.appendChild(levelCard);
    });
    
    levelDiv.appendChild(gridDiv);
    container.appendChild(levelDiv);
    
    addMethodDescription(container, 'Multi-level Cross-Validation',
        'Respects multiple hierarchical levels in medical data. Can validate at any level of the hierarchy.',
        ['Flexible validation level', 'Preserves hierarchical structure', 'Prevents multi-level leakage'],
        ['Complex implementation', 'Requires clear hierarchy definition', 'May have uneven splits']
    );
}

// Helper function for factorial calculation
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/**
 * Helper function for generating patient data
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