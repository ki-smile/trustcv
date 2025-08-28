/**
 * Spatial Cross-Validation Visualizations
 */

function visualizeSpatialBlock(nSplits) {
    const container = document.getElementById('cv-visualization');
    
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
        xaxis: {title: 'Longitude', showgrid: false},
        yaxis: {title: 'Latitude', showgrid: false},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
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
    
    // Show buffer zones concept
    const gridSize = 2;
    const data = [];
    
    // Test block
    data.push({
        x: [1, 2, 2, 1, 1],
        y: [1, 1, 2, 2, 1],
        fill: 'toself',
        fillcolor: colors.orange,
        name: 'Test Block',
        line: {color: 'white', width: 2}
    });
    
    // Buffer zone
    data.push({
        x: [0.5, 2.5, 2.5, 0.5, 0.5],
        y: [0.5, 0.5, 2.5, 2.5, 0.5],
        fill: 'toself',
        fillcolor: colors.grey,
        opacity: 0.3,
        name: 'Buffer Zone',
        line: {color: colors.grey, width: 1, dash: 'dash'}
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
            fillcolor: colors.plum,
            name: i === 0 ? 'Training Blocks' : '',
            showlegend: i === 0,
            line: {color: 'white', width: 1}
        });
    });
    
    const layout = {
        title: 'Buffered Spatial CV: Buffer Zones Reduce Autocorrelation',
        xaxis: {title: 'Longitude', showgrid: false, range: [-0.5, 3.5]},
        yaxis: {title: 'Latitude', showgrid: false, range: [-0.5, 3.5]},
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
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
                    colors_array.push(colors.orange);
                } else {
                    colors_array.push(colors.plum);
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
        scene: {
            xaxis: {title: 'Longitude'},
            yaxis: {title: 'Latitude'},
            zaxis: {title: 'Time'}
        },
        paper_bgcolor: colors.lightBlue
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
    
    // Show environmental factors integration
    const data = [
        {
            type: 'scatter',
            x: [1, 2, 3, 4, 5],
            y: [10, 15, 13, 17, 21],
            mode: 'markers+lines',
            name: 'Pollution Level',
            marker: {size: 10, color: colors.orange}
        },
        {
            type: 'scatter',
            x: [1, 2, 3, 4, 5],
            y: [5, 8, 6, 9, 12],
            mode: 'markers+lines',
            name: 'Disease Incidence',
            marker: {size: 10, color: colors.plum},
            yaxis: 'y2'
        }
    ];
    
    const layout = {
        title: 'Environmental Health CV: Multiple Covariates',
        xaxis: {title: 'Spatial Region'},
        yaxis: {title: 'Pollution (μg/m³)', side: 'left'},
        yaxis2: {
            title: 'Disease Rate (%)',
            overlaying: 'y',
            side: 'right'
        },
        paper_bgcolor: colors.lightBlue,
        plot_bgcolor: 'white'
    };
    
    Plotly.newPlot(container, data, layout, {responsive: true});
    
    addMethodDescription(container, 'Environmental Health Cross-Validation',
        'Specialized for environmental health studies with multiple factors.',
        ['Handles environmental covariates', 'Seasonal considerations'],
        ['Domain-specific']
    );
}