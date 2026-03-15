/**
 * Cross-Validation Animation for Homepage
 * Shows how k-fold CV works with interactive visualization
 */

class CVAnimation {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;
        
        // Configuration
        var tc = (typeof getThemeColors === 'function') ? getThemeColors() : {
            train: '#3498DB', test: '#E74C3C', inactive: '#BDC3C7',
            surface: '#FFFFFF', text: '#2C3E50', border: '#34495E', grey: '#7F8C8D'
        };
        this.config = {
            width: 500,
            height: 400,
            nSamples: 50,
            nFolds: 5,
            animationSpeed: 1500,
            colors: {
                train: tc.train,
                test: tc.test,
                inactive: tc.inactive,
                background: tc.surface,
                border: tc.text,
                text: tc.text,
                grey: tc.textMuted || tc.grey,
                foldLabel: tc.text
            }
        };
        
        this.currentFold = 0;
        this.isPlaying = false;
        this.animationTimer = null;
        
        this.init();
    }
    
    init() {
        // Create the visualization container
        this.container.innerHTML = `
            <div class="cv-animation-container" style="background: var(--md-sys-color-surface); border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div class="cv-header" style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: var(--ki-plum); margin: 0;">Cross-Validation in Action</h3>
                    <p style="color: var(--ki-grey); font-size: 14px; margin: 5px 0;">Watch how ${this.config.nFolds}-fold CV splits data</p>
                </div>
                
                <canvas id="cv-canvas" width="${this.config.width}" height="${this.config.height}" style="display: block; margin: 0 auto;"></canvas>
                
                <div class="cv-info" style="text-align: center; margin-top: 20px;">
                    <div class="fold-indicator" style="font-size: 18px; font-weight: bold; color: var(--ki-plum); margin-bottom: 10px;">
                        Fold <span id="current-fold">1</span> of ${this.config.nFolds}
                    </div>
                    <div class="legend" style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: 20px; height: 20px; background: ${this.config.colors.train}; border-radius: 4px;"></div>
                            <span style="font-size: 14px;">Training (${Math.floor(this.config.nSamples * (this.config.nFolds - 1) / this.config.nFolds)} samples)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: 20px; height: 20px; background: ${this.config.colors.test}; border-radius: 4px;"></div>
                            <span style="font-size: 14px;">Testing (${Math.floor(this.config.nSamples / this.config.nFolds)} samples)</span>
                        </div>
                    </div>
                </div>
                
                <div class="cv-controls" style="display: flex; justify-content: center; gap: 10px;">
                    <button id="cv-play" class="cv-btn" style="padding: 8px 20px; background: var(--ki-plum); color: white; border: none; border-radius: 20px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                        <span class="material-icons" style="font-size: 18px;">play_arrow</span>
                        Play
                    </button>
                    <button id="cv-reset" class="cv-btn" style="padding: 8px 20px; background: var(--ki-grey); color: var(--ki-dark-plum); border: none; border-radius: 20px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                        <span class="material-icons" style="font-size: 18px;">refresh</span>
                        Reset
                    </button>
                    <select id="cv-folds" style="padding: 8px 15px; border: 1px solid var(--border-color); border-radius: 20px; cursor: pointer;">
                        <option value="3">3-Fold</option>
                        <option value="5" selected>5-Fold</option>
                        <option value="10">10-Fold</option>
                    </select>
                </div>
                
                <div class="cv-explanation" style="margin-top: 20px; padding: 15px; background: var(--md-sys-color-surface-variant); border-radius: 8px; font-size: 14px; line-height: 1.6;">
                    <strong style="color: var(--ki-plum);">How it works:</strong><br>
                    • Data is split into <strong>${this.config.nFolds} equal folds</strong><br>
                    • Each fold serves as test set once while others train<br>
                    • Every sample is tested exactly once<br>
                    • Final score = average across all folds
                </div>
            </div>
        `;
        
        this.canvas = document.getElementById('cv-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.setupEventListeners();
        this.generateData();
        this.draw();
    }
    
    setupEventListeners() {
        const playBtn = document.getElementById('cv-play');
        const resetBtn = document.getElementById('cv-reset');
        const foldsSelect = document.getElementById('cv-folds');
        
        if (playBtn) {
            playBtn.addEventListener('click', () => this.togglePlay());
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.reset());
        }
        
        if (foldsSelect) {
            foldsSelect.addEventListener('change', (e) => {
                this.config.nFolds = parseInt(e.target.value);
                this.reset();
                this.updateInfo();
            });
        }
    }
    
    generateData() {
        this.data = [];
        const samplesPerFold = Math.floor(this.config.nSamples / this.config.nFolds);
        
        for (let i = 0; i < this.config.nSamples; i++) {
            this.data.push({
                id: i,
                fold: Math.floor(i / samplesPerFold),
                x: 0,
                y: 0
            });
        }
    }
    
    draw() {
        if (!this.ctx) return;
        
        // Clear canvas
        this.ctx.fillStyle = this.config.colors.background;
        this.ctx.fillRect(0, 0, this.config.width, this.config.height);
        
        const paddingLeft = 65;
        const padding = 20;
        const availableWidth = this.config.width - paddingLeft - padding;
        const availableHeight = this.config.height - 2 * padding;
        
        // Calculate grid layout
        const cols = Math.ceil(Math.sqrt(this.config.nSamples));
        const rows = Math.ceil(this.config.nSamples / cols);
        const cellWidth = availableWidth / cols;
        const cellHeight = availableHeight / rows;
        const dotRadius = Math.min(cellWidth, cellHeight) * 0.3;
        
        // Draw fold boundaries
        const samplesPerFold = Math.floor(this.config.nSamples / this.config.nFolds);
        
        for (let fold = 1; fold < this.config.nFolds; fold++) {
            const boundaryIndex = fold * samplesPerFold;
            const boundaryCol = boundaryIndex % cols;
            const boundaryRow = Math.floor(boundaryIndex / cols);
            
            if (boundaryCol === 0 && boundaryRow > 0) {
                // Horizontal line between rows
                this.ctx.strokeStyle = this.config.colors.inactive;
                this.ctx.lineWidth = 2;
                this.ctx.setLineDash([5, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(paddingLeft, padding + boundaryRow * cellHeight);
                this.ctx.lineTo(paddingLeft + availableWidth, padding + boundaryRow * cellHeight);
                this.ctx.stroke();
                this.ctx.setLineDash([]);
            }
        }
        
        // Draw data points
        this.data.forEach((point, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            const x = paddingLeft + col * cellWidth + cellWidth / 2;
            const y = padding + row * cellHeight + cellHeight / 2;
            
            // Determine color based on current fold
            let color = this.config.colors.inactive;
            if (point.fold === this.currentFold) {
                color = this.config.colors.test;
            } else {
                color = this.config.colors.train;
            }
            
            // Draw point
            this.ctx.beginPath();
            this.ctx.arc(x, y, dotRadius, 0, 2 * Math.PI);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            
            // Add border
            this.ctx.strokeStyle = this.config.colors.border;
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
            
            // Animate test fold
            if (point.fold === this.currentFold) {
                // Add pulsing effect for test samples
                const time = Date.now() / 1000;
                const scale = 1 + Math.sin(time * 3) * 0.1;
                
                this.ctx.beginPath();
                this.ctx.arc(x, y, dotRadius * scale, 0, 2 * Math.PI);
                this.ctx.strokeStyle = this.config.colors.test;
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
            }
        });
        
        // Draw fold labels in the left margin
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'middle';

        for (let fold = 0; fold < this.config.nFolds; fold++) {
            const startIdx = fold * samplesPerFold;
            const endIdx = Math.min(startIdx + samplesPerFold - 1, this.config.nSamples - 1);
            // Find the vertical center of this fold's rows
            const startRow = Math.floor(startIdx / cols);
            const endRow = Math.floor(endIdx / cols);
            const midRow = (startRow + endRow) / 2;
            const y = padding + midRow * cellHeight + cellHeight / 2;
            const x = paddingLeft - 8;

            if (fold === this.currentFold) {
                // Draw highlighted TEST label with background pill
                var label = 'TEST';
                this.ctx.font = 'bold 13px -apple-system, BlinkMacSystemFont, sans-serif';
                var tw = this.ctx.measureText(label).width;
                // Pill background
                this.ctx.fillStyle = this.config.colors.test;
                var pillH = 20, pillW = tw + 14, px = x - pillW, py = y - pillH / 2;
                this.ctx.beginPath();
                if (this.ctx.roundRect) {
                    this.ctx.roundRect(px, py, pillW, pillH, 10);
                } else {
                    this.ctx.rect(px, py, pillW, pillH);
                }
                this.ctx.fill();
                // Text
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillText(label, x - 7, y);
            } else {
                // Draw subtle fold label
                this.ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
                this.ctx.fillStyle = this.config.colors.foldLabel;
                this.ctx.globalAlpha = 0.6;
                this.ctx.fillText(`F${fold + 1}`, x - 4, y);
                this.ctx.globalAlpha = 1.0;
            }
        }
        this.ctx.textBaseline = 'alphabetic';
        
        // Request next frame for smooth animation
        if (this.isPlaying) {
            requestAnimationFrame(() => this.draw());
        }
    }
    
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const playBtn = document.getElementById('cv-play');
        
        if (this.isPlaying) {
            playBtn.innerHTML = `
                <span class="material-icons" style="font-size: 18px;">pause</span>
                Pause
            `;
            this.startAnimation();
        } else {
            playBtn.innerHTML = `
                <span class="material-icons" style="font-size: 18px;">play_arrow</span>
                Play
            `;
            this.stopAnimation();
        }
    }
    
    startAnimation() {
        this.animationTimer = setInterval(() => {
            this.currentFold = (this.currentFold + 1) % this.config.nFolds;
            this.updateFoldIndicator();
            this.draw();
        }, this.config.animationSpeed);
        
        // Start drawing loop
        this.draw();
    }
    
    stopAnimation() {
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }
    }
    
    reset() {
        this.stopAnimation();
        this.isPlaying = false;
        this.currentFold = 0;
        
        const playBtn = document.getElementById('cv-play');
        if (playBtn) {
            playBtn.innerHTML = `
                <span class="material-icons" style="font-size: 18px;">play_arrow</span>
                Play
            `;
        }
        
        this.updateFoldIndicator();
        this.generateData();
        this.draw();
    }
    
    updateFoldIndicator() {
        const indicator = document.getElementById('current-fold');
        if (indicator) {
            indicator.textContent = this.currentFold + 1;
        }
    }
    
    updateInfo() {
        // Update the explanation text
        const explanation = this.container.querySelector('.cv-explanation');
        if (explanation) {
            explanation.innerHTML = `
                <strong style="color: var(--ki-plum);">How it works:</strong><br>
                • Data is split into <strong>${this.config.nFolds} equal folds</strong><br>
                • Each fold serves as test set once while others train<br>
                • Every sample is tested exactly once<br>
                • Final score = average across all folds
            `;
        }
        
        // Update legend
        const trainSamples = Math.floor(this.config.nSamples * (this.config.nFolds - 1) / this.config.nFolds);
        const testSamples = Math.floor(this.config.nSamples / this.config.nFolds);
        
        const legend = this.container.querySelector('.legend');
        if (legend) {
            legend.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background: ${this.config.colors.train}; border-radius: 4px;"></div>
                    <span style="font-size: 14px;">Training (${trainSamples} samples)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background: ${this.config.colors.test}; border-radius: 4px;"></div>
                    <span style="font-size: 14px;">Testing (${testSamples} samples)</span>
                </div>
            `;
        }
    }
}

// Initialize animation when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('cv-demo-plot');
    if (container) {
        window.cvHomepageAnimation = new CVAnimation('cv-demo-plot');
    }
});

// Re-initialize on theme change so colors update
window.addEventListener('themechange', function() {
    var container = document.getElementById('cv-demo-plot');
    if (container) {
        window.cvHomepageAnimation = new CVAnimation('cv-demo-plot');
    }
});