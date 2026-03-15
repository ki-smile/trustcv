/**
 * Spatial Cross-Validation Visualizations
 */

function getColors() {
    return (typeof getThemeColors === 'function') ? getThemeColors() : { plum:'#870052', darkPlum:'#4F0433', orange:'#FF876F', lightBlue:'#EDF4F4', grey:'#6B6B6B', text:'#000000', plotBg:'#FFFFFF', paperBg:'#EDF4F4', axisColor:'#666666', gridColor:'rgba(0,0,0,0.06)', train:'#3498DB', test:'#E74C3C', inactive:'#BDC3C7', surface:'#FFFFFF', border:'#EDDBE4' };
}

function visualizeSpatialBlock(nSplits) {
    const container = document.getElementById('cv-visualization');
    const c = getColors();

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
                    fillcolor: blockId === 0 ? c.orange : c.plum,
                    line: {color: c.border, width: 2},
                    name: `Block ${blockId + 1}`,
                    showlegend: blockId < 2
                });
            }
        }
    }

    const layout = {
        title: 'Spatial Block Cross-Validation',
        font: { color: c.text },
        xaxis: {title: 'Longitude', showgrid: false, color: c.axisColor, gridcolor: c.gridColor},
        yaxis: {title: 'Latitude', showgrid: false, color: c.axisColor, gridcolor: c.gridColor},
        paper_bgcolor: c.paperBg,
        plot_bgcolor: c.plotBg
    };

    Plotly.newPlot(container, data, layout, {responsive: true});

    addMethodDescription(container, 'Spatial Block Cross-Validation',
        'Divides geographic space into blocks for validation.',
        ['Handles spatial autocorrelation', 'Tests spatial generalization'],
        ['May have uneven sample distribution']
    );
}

function visualizeBufferedSpatial(nSplits) {
    const container = document.getElementById('cv-visualization');
    const c = getColors();

    // Show buffer zones concept
    const gridSize = 2;
    const data = [];

    // Test block
    data.push({
        x: [1, 2, 2, 1, 1],
        y: [1, 1, 2, 2, 1],
        fill: 'toself',
        fillcolor: c.orange,
        name: 'Test Block',
        line: {color: c.border, width: 2}
    });

    // Buffer zone
    data.push({
        x: [0.5, 2.5, 2.5, 0.5, 0.5],
        y: [0.5, 0.5, 2.5, 2.5, 0.5],
        fill: 'toself',
        fillcolor: c.grey,
        opacity: 0.3,
        name: 'Buffer Zone',
        line: {color: c.grey, width: 1, dash: 'dash'}
    });

    // Training blocks
    const trainingBlocks = [
        {x: [0, 0.4], y: [0, 0.4]},
        {x: [2.6, 3], y: [0, 0.4]},
        {x: [0, 0.4], y: [2.6, 3]},
        {x: [2.6, 3], y: [2.6, 3]}
    ];

    trainingBlocks.forEach((block, i) => {
        data.push({
            x: [block.x[0], block.x[1], block.x[1], block.x[0], block.x[0]],
            y: [block.y[0], block.y[0], block.y[1], block.y[1], block.y[0]],
            fill: 'toself',
            fillcolor: c.plum,
            name: i === 0 ? 'Training Blocks' : '',
            showlegend: i === 0,
            line: {color: c.border, width: 1}
        });
    });

    const layout = {
        title: 'Buffered Spatial CV: Buffer Zones Reduce Autocorrelation',
        font: { color: c.text },
        xaxis: {title: 'Longitude', showgrid: false, range: [-0.5, 3.5], color: c.axisColor, gridcolor: c.gridColor},
        yaxis: {title: 'Latitude', showgrid: false, range: [-0.5, 3.5], color: c.axisColor, gridcolor: c.gridColor},
        paper_bgcolor: c.paperBg,
        plot_bgcolor: c.plotBg
    };

    Plotly.newPlot(container, data, layout, {responsive: true});

    addMethodDescription(container, 'Buffered Spatial Cross-Validation',
        'Adds buffer zones around test blocks to reduce autocorrelation.',
        ['Stronger independence', 'Better generalization test'],
        ['Loses training data in buffers']
    );
}

function visualizeSpatiotemporal() {
    const container = document.getElementById('cv-visualization');
    const c = getColors();

    // Create 3D visualization for spatiotemporal
    const x = [], y = [], z = [], colors_array = [];

    // Generate spatiotemporal blocks
    for (let t = 0; t < 4; t++) { // Time blocks
        for (let i = 0; i < 3; i++) { // Spatial X
            for (let j = 0; j < 3; j++) { // Spatial Y
                x.push(i);
                y.push(j);
                z.push(t);
                // Color one block as test
                if (t === 2 && i === 1 && j === 1) {
                    colors_array.push(c.orange);
                } else {
                    colors_array.push(c.plum);
                }
            }
        }
    }

    const data = [{
        x: x,
        y: y,
        z: z,
        mode: 'markers',
        marker: {
            size: 12,
            color: colors_array,
            opacity: 0.8
        },
        type: 'scatter3d'
    }];

    const layout = {
        title: 'Spatiotemporal Block CV: Space + Time Dimensions',
        font: { color: c.text },
        scene: {
            xaxis: {title: 'Longitude', color: c.axisColor, gridcolor: c.gridColor},
            yaxis: {title: 'Latitude', color: c.axisColor, gridcolor: c.gridColor},
            zaxis: {title: 'Time', color: c.axisColor, gridcolor: c.gridColor}
        },
        paper_bgcolor: c.paperBg
    };

    Plotly.newPlot(container, data, layout, {responsive: true});

    addMethodDescription(container, 'Spatiotemporal Block Cross-Validation',
        'Combines spatial and temporal blocking for 4D data.',
        ['Handles space-time dependencies', 'Good for disease spread'],
        ['Complex block structure']
    );
}

function visualizeEnvironmental() {
    const container = document.getElementById('cv-visualization');
    const c = getColors();

    // Show environmental factors integration
    const data = [
        {
            type: 'scatter',
            x: [1, 2, 3, 4, 5],
            y: [10, 15, 13, 17, 21],
            mode: 'markers+lines',
            name: 'Pollution Level',
            marker: {size: 10, color: c.orange}
        },
        {
            type: 'scatter',
            x: [1, 2, 3, 4, 5],
            y: [5, 8, 6, 9, 12],
            mode: 'markers+lines',
            name: 'Disease Incidence',
            marker: {size: 10, color: c.plum},
            yaxis: 'y2'
        }
    ];

    const layout = {
        title: 'Environmental Health CV: Multiple Covariates',
        font: { color: c.text },
        xaxis: {title: 'Spatial Region', color: c.axisColor, gridcolor: c.gridColor},
        yaxis: {title: 'Pollution (μg/m³)', side: 'left', color: c.axisColor, gridcolor: c.gridColor},
        yaxis2: {
            title: 'Disease Rate (%)',
            overlaying: 'y',
            side: 'right',
            color: c.axisColor,
            gridcolor: c.gridColor
        },
        paper_bgcolor: c.paperBg,
        plot_bgcolor: c.plotBg
    };

    Plotly.newPlot(container, data, layout, {responsive: true});

    addMethodDescription(container, 'Environmental Health Cross-Validation',
        'Specialized for environmental health studies with multiple factors.',
        ['Handles environmental covariates', 'Seasonal considerations'],
        ['Domain-specific']
    );
}

window.addEventListener('themechange', function() {
    try { if(typeof updateVisualization === 'function') updateVisualization(); } catch(e) {}
});
