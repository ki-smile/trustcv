/**
 * Temporal Cross-Validation Visualizations
 */

function getColors() {
    return (typeof getThemeColors === 'function') ? getThemeColors() : { plum:'#870052', darkPlum:'#4F0433', orange:'#FF876F', lightBlue:'#EDF4F4', grey:'#6B6B6B', text:'#000000', plotBg:'#FFFFFF', paperBg:'#EDF4F4', axisColor:'#666666', gridColor:'rgba(0,0,0,0.06)', titleColor:'#4F0433', annotationColor:'#666666', train:'#3498DB', test:'#E74C3C', inactive:'#BDC3C7' };
}

function visualizeTemporal(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    const dates = [];
    const values = [];
    const n = 100;
    
    for (let i = 0; i < n; i++) {
        dates.push(new Date(2024, 0, i + 1));
        values.push(Math.sin(i / 10) * 20 + 50 + Math.random() * 10);
    }
    
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
            fillcolor: fold === nSplits - 1 ? getColors().orange : getColors().plum,
            opacity: 0.2,
            line: {width: 0}
        });
        
        annotations.push({
            x: dates[start + foldSize / 2],
            y: 75,
            text: fold === nSplits - 1 ? 'Test' : `Train ${fold + 1}`,
            showarrow: false,
            font: {color: getColors().annotationColor}
        });
    }
    
    const trace = {
        x: dates,
        y: values,
        type: 'scatter',
        mode: 'lines',
        name: 'Patient Vital Signs',
        line: {color: getColors().darkPlum, width: 2}
    };
    
    const layout = {
        title: 'Time Series Split: Respects Temporal Order',
        font: {color: getColors().text},
        xaxis: {title: 'Time', color: getColors().axisColor, gridcolor: getColors().gridColor},
        yaxis: {title: 'Measurement Value', color: getColors().axisColor, gridcolor: getColors().gridColor},
        shapes: shapes,
        annotations: annotations,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    addMethodDescription(container, 'Time Series Cross-Validation',
        'For temporal medical data: always train on past, test on future.',
        ['Prevents temporal leakage', 'Realistic for predictive models', 'Suitable for patient monitoring'],
        ['Less data for early folds']
    );
}

function visualizeRollingWindow() {
    const container = document.getElementById('cv-visualization');
    
    // Create rolling window visualization
    const n = 50;
    const windowSize = 15;
    const stepSize = 5;
    const forecastHorizon = 5;
    
    const traces = [];
    const shapes = [];
    
    for (let i = 0; i < 3; i++) {
        const start = i * stepSize;
        const trainEnd = start + windowSize;
        const testEnd = trainEnd + forecastHorizon;
        
        shapes.push({
            type: 'rect',
            x0: start,
            x1: trainEnd,
            y0: i * 0.3,
            y1: i * 0.3 + 0.25,
            fillcolor: getColors().plum,
            opacity: 0.5,
            line: {width: 0}
        });
        
        shapes.push({
            type: 'rect',
            x0: trainEnd,
            x1: testEnd,
            y0: i * 0.3,
            y1: i * 0.3 + 0.25,
            fillcolor: getColors().orange,
            opacity: 0.5,
            line: {width: 0}
        });
    }
    
    const layout = {
        title: 'Rolling Window: Fixed-Size Training Window',
        font: {color: getColors().text},
        xaxis: {title: 'Time Index', range: [0, 40], color: getColors().axisColor, gridcolor: getColors().gridColor},
        yaxis: {title: 'Window', range: [-0.1, 1], showticklabels: false, color: getColors().axisColor, gridcolor: getColors().gridColor},
        shapes: shapes,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg,
        showlegend: false
    };
    
    Plotly.newPlot(container, [], layout, {responsive: true});
    
    addMethodDescription(container, 'Rolling Window Cross-Validation',
        'Fixed-size training window slides through time.',
        ['Constant training size', 'Good for stationary series'],
        ['Discards old data']
    );
}

function visualizeExpandingWindow() {
    const container = document.getElementById('cv-visualization');
    
    const shapes = [];
    const minTrainSize = 10;
    const stepSize = 10;
    const forecastHorizon = 5;
    
    for (let i = 0; i < 3; i++) {
        const trainEnd = minTrainSize + i * stepSize;
        const testEnd = trainEnd + forecastHorizon;
        
        shapes.push({
            type: 'rect',
            x0: 0,
            x1: trainEnd,
            y0: i * 0.3,
            y1: i * 0.3 + 0.25,
            fillcolor: getColors().plum,
            opacity: 0.5,
            line: {width: 0}
        });
        
        shapes.push({
            type: 'rect',
            x0: trainEnd,
            x1: testEnd,
            y0: i * 0.3,
            y1: i * 0.3 + 0.25,
            fillcolor: getColors().orange,
            opacity: 0.5,
            line: {width: 0}
        });
    }
    
    const layout = {
        title: 'Expanding Window: Growing Training Set',
        font: {color: getColors().text},
        xaxis: {title: 'Time Index', range: [0, 40], color: getColors().axisColor, gridcolor: getColors().gridColor},
        yaxis: {title: 'Window', range: [-0.1, 1], showticklabels: false, color: getColors().axisColor, gridcolor: getColors().gridColor},
        shapes: shapes,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg,
        showlegend: false
    };
    
    Plotly.newPlot(container, [], layout, {responsive: true});
    
    addMethodDescription(container, 'Expanding Window Cross-Validation',
        'Training set grows over time, always starting from beginning.',
        ['Uses all historical data', 'Good for learning curves'],
        ['Varying training size']
    );
}

function visualizeBlockedTimeSeries(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    // Create visualization showing temporal blocks
    const daysPerBlock = 7; // Weekly blocks
    const totalDays = daysPerBlock * nSplits * 2; // Some extra for visualization
    const dates = [];
    const values = [];
    const blockColors = [];
    
    // Generate time series data with weekly patterns
    for (let i = 0; i < totalDays; i++) {
        dates.push(new Date(2024, 0, i + 1));
        // Add weekly seasonality
        values.push(50 + 10 * Math.sin(2 * Math.PI * i / 7) + Math.random() * 5);
        
        // Assign block colors
        const blockId = Math.floor(i / daysPerBlock);
        if (blockId % nSplits === nSplits - 1) {
            blockColors.push(getColors().orange); // Test blocks
        } else {
            blockColors.push(getColors().plum); // Train blocks
        }
    }
    
    // Create shapes for blocks
    const shapes = [];
    const annotations = [];
    
    for (let block = 0; block < Math.floor(totalDays / daysPerBlock); block++) {
        const startIdx = block * daysPerBlock;
        const endIdx = Math.min(startIdx + daysPerBlock, totalDays - 1);
        
        const isTestBlock = block % nSplits === nSplits - 1;
        
        shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: dates[startIdx],
            x1: dates[endIdx],
            y0: 0,
            y1: 1,
            fillcolor: isTestBlock ? getColors().orange : getColors().plum,
            opacity: 0.2,
            line: {width: 1, color: isTestBlock ? getColors().orange : getColors().plum}
        });
        
        if (block < nSplits * 2) { // Only label first few blocks
            annotations.push({
                x: dates[startIdx + Math.floor(daysPerBlock / 2)],
                y: 35,
                text: `Week ${block + 1}<br>${isTestBlock ? 'TEST' : 'TRAIN'}`,
                showarrow: false,
                font: {size: 10, color: getColors().annotationColor}
            });
        }
    }

    const trace = {
        x: dates,
        y: values,
        type: 'scatter',
        mode: 'lines',
        name: 'Patient Monitoring Data',
        line: {color: getColors().darkPlum, width: 2}
    };
    
    const layout = {
        title: 'Blocked Time Series: Weekly Blocks Stay Together',
        font: {color: getColors().text},
        xaxis: {
            title: 'Date',
            tickformat: '%b %d',
            color: getColors().axisColor,
            gridcolor: getColors().gridColor
        },
        yaxis: {title: 'Measurement Value', color: getColors().axisColor, gridcolor: getColors().gridColor},
        shapes: shapes,
        annotations: annotations,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg,
        hovermode: 'x unified'
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
    
    addMethodDescription(container, 'Blocked Time Series Cross-Validation',
        'Preserves temporal blocks (days, weeks, months) together to maintain temporal patterns.',
        ['Handles seasonal patterns', 'Preserves temporal structure', 'Reduces data leakage within blocks'],
        ['May have uneven splits', 'Block size selection is critical']
    );
}

function visualizePurgedKFold(nSplits) {
    const container = document.getElementById('cv-visualization');
    
    // Show purge gaps
    const n = 50;
    const foldSize = Math.floor(n / nSplits);
    const purgeGap = 3;
    
    const shapes = [];
    const colors_map = {
        train: getColors().plum,
        test: getColors().orange,
        purge: getColors().grey
    };
    
    for (let fold = 0; fold < nSplits; fold++) {
        const testStart = fold * foldSize;
        const testEnd = testStart + foldSize;
        
        // Test fold
        shapes.push({
            type: 'rect',
            x0: testStart,
            x1: testEnd,
            y0: 0.4,
            y1: 0.6,
            fillcolor: colors_map.test,
            opacity: 0.7,
            line: {width: 0}
        });
        
        // Purge before
        if (testStart > 0) {
            shapes.push({
                type: 'rect',
                x0: Math.max(0, testStart - purgeGap),
                x1: testStart,
                y0: 0.4,
                y1: 0.6,
                fillcolor: colors_map.purge,
                opacity: 0.3,
                line: {width: 0}
            });
        }
        
        // Purge after
        if (testEnd < n) {
            shapes.push({
                type: 'rect',
                x0: testEnd,
                x1: Math.min(n, testEnd + purgeGap),
                y0: 0.4,
                y1: 0.6,
                fillcolor: colors_map.purge,
                opacity: 0.3,
                line: {width: 0}
            });
        }
    }
    
    const layout = {
        title: 'Purged K-Fold: Gaps Prevent Information Leakage',
        font: {color: getColors().text},
        xaxis: {title: 'Sample Index', range: [0, n], color: getColors().axisColor, gridcolor: getColors().gridColor},
        yaxis: {title: '', range: [0, 1], showticklabels: false, color: getColors().axisColor, gridcolor: getColors().gridColor},
        shapes: shapes,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg
    };
    
    Plotly.newPlot(container, [], layout, {responsive: true});
    
    addMethodDescription(container, 'Purged K-Fold Cross-Validation',
        'K-fold with temporal gap (purge) between train and test to prevent leakage.',
        ['Prevents information leakage', 'Good for financial data'],
        ['Reduces training data']
    );
}

function visualizeCPCV() {
    const container = document.getElementById('cv-visualization');
    
    // Create visualization showing combinatorial nature
    const n = 50;
    const nGroups = 5;
    const nTestGroups = 2;
    const groupSize = Math.floor(n / nGroups);
    const purgeGap = 2;
    
    // Create multiple combination examples
    const combinations = [
        [0, 1], [0, 2], [1, 3], [2, 4] // Sample combinations
    ];
    
    const traces = [];
    const shapes = [];
    
    // Create a subplot for each combination
    combinations.forEach((combo, idx) => {
        const yBase = idx * 0.2;
        
        for (let g = 0; g < nGroups; g++) {
            const xStart = g * groupSize;
            const xEnd = xStart + groupSize;
            
            let color, opacity;
            if (combo.includes(g)) {
                // Test group
                color = getColors().orange;
                opacity = 0.8;
            } else {
                // Check if in purge zone
                let inPurge = false;
                combo.forEach(testG => {
                    const testStart = testG * groupSize;
                    const testEnd = testStart + groupSize;
                    if (Math.abs(xStart - testEnd) <= purgeGap || 
                        Math.abs(xEnd - testStart) <= purgeGap) {
                        inPurge = true;
                    }
                });
                
                if (inPurge) {
                    color = getColors().grey;
                    opacity = 0.3;
                } else {
                    color = getColors().plum;
                    opacity = 0.6;
                }
            }
            
            shapes.push({
                type: 'rect',
                x0: xStart,
                x1: xEnd - 0.5,
                y0: yBase,
                y1: yBase + 0.15,
                fillcolor: color,
                opacity: opacity,
                line: {width: 1, color: getColors().plotBg}
            });
        }
        
        // Add combination label
        traces.push({
            x: [n / 2],
            y: [yBase + 0.075],
            text: `C(${combo[0]},${combo[1]})`,
            mode: 'text',
            showlegend: false,
            textfont: {size: 10, color: getColors().darkPlum}
        });
    });
    
    const layout = {
        title: 'Combinatorial Purged CV: Multiple Test Group Combinations',
        font: {color: getColors().text},
        xaxis: {
            title: 'Sample Index',
            range: [0, n],
            showgrid: false,
            color: getColors().axisColor,
            gridcolor: getColors().gridColor
        },
        yaxis: {
            title: 'Combinations',
            range: [-0.05, 0.85],
            showticklabels: false,
            showgrid: false,
            color: getColors().axisColor,
            gridcolor: getColors().gridColor
        },
        shapes: shapes,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg,
        showlegend: false,
        annotations: [
            {
                x: 0.5,
                y: -0.15,
                xref: 'paper',
                yref: 'paper',
                text: 'Orange: Test | Grey: Purged | Purple: Train',
                showarrow: false,
                font: {size: 11, color: getColors().annotationColor}
            }
        ]
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true});
    
    // Add combination formula
    const formulaDiv = document.createElement('div');
    formulaDiv.style.marginTop = '15px';
    formulaDiv.style.padding = '10px';
    formulaDiv.style.backgroundColor = getColors().plotBg;
    formulaDiv.style.borderRadius = '8px';
    formulaDiv.innerHTML = `
        <h4 style="color: ${getColors().titleColor};">Total Combinations: C(${nGroups}, ${nTestGroups}) = ${combination(nGroups, nTestGroups)}</h4>
        <p style="color: ${getColors().grey}; font-size: 14px;">
            Each combination tests different group pairs with purging to prevent leakage.
        </p>
    `;
    container.appendChild(formulaDiv);
    
    addMethodDescription(container, 'Combinatorial Purged Cross-Validation',
        'Generates all possible combinations of test groups with temporal purging.',
        ['Robust validation through many scenarios', 'Comprehensive testing', 'Prevents all forms of leakage'],
        ['Computationally intensive', 'Many iterations to run']
    );
}

// Helper function for combination calculation
function combination(n, k) {
    if (k > n) return 0;
    if (k === 0 || k === n) return 1;
    
    let result = 1;
    for (let i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
}

function visualizePurgedGroupTimeSeries() {
    showTextDescription('Purged Group Time Series Split',
        'Combines temporal order, patient grouping, and purging.',
        '',
        ['Handles panel data', 'Prevents all leakage types'],
        ['Complex to implement']
    );
}

function visualizeNestedTemporal() {
    const container = document.getElementById('cv-visualization');
    
    // Create nested visualization showing outer and inner loops
    const outerFolds = 3;
    const innerFolds = 3;
    
    const data = [];
    const shapes = [];
    const annotations = [];
    
    // Create timeline
    const totalSamples = 100;
    const timePoints = Array.from({length: totalSamples}, (_, i) => i);
    
    // Outer loop visualization
    const outerHeight = 0.6;
    const innerHeight = 0.3;
    
    for (let outer = 0; outer < outerFolds; outer++) {
        const outerTestStart = 60 + outer * 15;
        const outerTestEnd = outerTestStart + 15;
        const outerTrainEnd = outerTestStart - 5; // Gap
        
        // Outer test fold
        shapes.push({
            type: 'rect',
            x0: outerTestStart,
            x1: outerTestEnd,
            y0: outerHeight,
            y1: outerHeight + 0.15,
            fillcolor: getColors().orange,
            opacity: 0.7,
            line: {width: 1, color: getColors().orange}
        });
        
        // Outer train region
        shapes.push({
            type: 'rect',
            x0: 0,
            x1: outerTrainEnd,
            y0: outerHeight,
            y1: outerHeight + 0.15,
            fillcolor: getColors().plum,
            opacity: 0.3,
            line: {width: 1, color: getColors().plum}
        });
        
        // Add outer fold label
        annotations.push({
            x: outerTestStart + 7,
            y: outerHeight + 0.2,
            text: `Outer ${outer + 1}`,
            showarrow: false,
            font: {size: 10, color: getColors().darkPlum}
        });
        
        // Inner loop visualization (show for first outer fold only)
        if (outer === 0) {
            const innerTrainSize = outerTrainEnd;
            
            for (let inner = 0; inner < innerFolds; inner++) {
                const innerTestStart = 15 + inner * 15;
                const innerTestEnd = innerTestStart + 10;
                
                if (innerTestEnd <= outerTrainEnd) {
                    // Inner test fold
                    shapes.push({
                        type: 'rect',
                        x0: innerTestStart,
                        x1: innerTestEnd,
                        y0: innerHeight,
                        y1: innerHeight + 0.1,
                        fillcolor: getColors().orange,
                        opacity: 0.5,
                        line: {width: 0.5, color: getColors().orange}
                    });
                    
                    // Inner train region
                    shapes.push({
                        type: 'rect',
                        x0: 0,
                        x1: innerTestStart - 2,
                        y0: innerHeight,
                        y1: innerHeight + 0.1,
                        fillcolor: getColors().plum,
                        opacity: 0.2,
                        line: {width: 0.5, color: getColors().plum}
                    });
                    
                    // Inner fold label
                    annotations.push({
                        x: innerTestStart + 5,
                        y: innerHeight - 0.05,
                        text: `Inner ${inner + 1}`,
                        showarrow: false,
                        font: {size: 8, color: getColors().grey}
                    });
                }
            }
        }
    }
    
    // Add legend annotations
    annotations.push({
        x: 10,
        y: 0.9,
        text: 'OUTER LOOP (Model Evaluation)',
        showarrow: false,
        font: {size: 12, color: getColors().darkPlum, family: 'Arial Black'}
    });
    
    annotations.push({
        x: 10,
        y: 0.15,
        text: 'INNER LOOP (Hyperparameter Tuning)',
        showarrow: false,
        font: {size: 10, color: getColors().grey}
    });
    
    const layout = {
        title: 'Nested Temporal CV: Two-Level Time-Aware Validation',
        font: {color: getColors().text},
        xaxis: {
            title: 'Time Index',
            range: [0, totalSamples],
            showgrid: true,
            gridcolor: getColors().gridColor,
            color: getColors().axisColor
        },
        yaxis: {
            title: '',
            range: [0, 1],
            showticklabels: false,
            showgrid: false,
            color: getColors().axisColor,
            gridcolor: getColors().gridColor
        },
        shapes: shapes,
        annotations: annotations,
        paper_bgcolor: getColors().paperBg,
        plot_bgcolor: getColors().plotBg,
        showlegend: false,
        margin: {t: 50, b: 50, l: 50, r: 50}
    };
    
    Plotly.newPlot(container, [], layout, {responsive: true});
    
    // Add explanation
    const explanationDiv = document.createElement('div');
    explanationDiv.style.marginTop = '15px';
    explanationDiv.style.padding = '15px';
    explanationDiv.style.backgroundColor = getColors().plotBg;
    explanationDiv.style.borderRadius = '8px';
    explanationDiv.innerHTML = `
        <h4 style="color: ${getColors().titleColor};">How Nested Temporal CV Works:</h4>
        <ol style="color: ${getColors().grey}; font-size: 14px; line-height: 1.6;">
            <li><strong>Outer Loop:</strong> Evaluates final model performance (orange = test, purple = train)</li>
            <li><strong>Inner Loop:</strong> Finds best hyperparameters using only outer training data</li>
            <li><strong>Temporal Constraint:</strong> Both loops respect time order - no future data leakage</li>
            <li><strong>Result:</strong> Unbiased performance estimate with optimized hyperparameters</li>
        </ol>
    `;
    container.appendChild(explanationDiv);
    
    addMethodDescription(container, 'Nested Temporal Cross-Validation',
        'Two-level validation preserving temporal order in both loops for unbiased hyperparameter tuning.',
        ['Proper time series hyperparameter optimization', 'No future information leak', 'Unbiased performance estimates'],
        ['Very computationally expensive', 'Complex to implement correctly', 'Requires large datasets']
    );
}

window.addEventListener('themechange', function() {
    var methodSelect = document.getElementById('cv-method');
    if (methodSelect) {
        try { updateVisualization(); } catch(e) {}
    }
});