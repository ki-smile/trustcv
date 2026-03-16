// Theme toggle for trustcv website
// Dark theme is default

function toggleTheme() {
    var html = document.documentElement;
    var current = html.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('trustcv-theme', next);
    updateThemeIcon(next);
    // Dispatch event so visualizations can re-render
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
}

function updateThemeIcon(theme) {
    var icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = theme === 'dark' ? 'light_mode' : 'dark_mode';
}

// Returns theme-aware colors for JS visualizations (Plotly, Canvas, etc.)
// Dynamically reads from CSS variables defined in style.css
function getThemeColors() {
    const style = getComputedStyle(document.documentElement);
    
    const getVar = (name, fallback) => {
        const val = style.getPropertyValue(name).trim();
        return val || fallback;
    };

    return {
        // Backgrounds
        background: getVar('--md-sys-color-background', '#12010F'),
        surface: getVar('--md-sys-color-surface', '#1A0118'),
        cardBg: getVar('--card-bg', '#1F1030'),
        heroBg: getVar('--section-dark', '#0E000C'),
        plotBg: getVar('--md-sys-color-surface', '#1A0118'),
        paperBg: getVar('--md-sys-color-background', '#12010F'),
        
        // Text
        text: getVar('--ki-dark-plum', '#F5EDF3'),
        textMuted: getVar('--ki-grey', '#A08898'),
        
        // Accents
        plum: getVar('--ki-plum', '#C066A0'),
        darkPlum: getVar('--ki-brand-dark-plum', '#4F0433'),
        orange: getVar('--ki-orange', '#FF876F'),
        lightBlue: getVar('--ki-light-blue', '#1F1030'),
        lightOrange: getVar('--ki-light-orange', '#240820'),
        
        // Functional
        train: '#5B9BD5', // Constant functional colors can stay or be variables
        test: '#FF6B6B',
        validate: '#FFB347',
        inactive: getVar('--ki-grey', '#5A3D6A'),
        border: getVar('--border-color', 'rgba(255,135,111,0.1)'),
        grey: getVar('--ki-grey', '#A08898'),
        gridColor: 'rgba(255,135,111,0.08)',
        
        // Chart specific
        axisColor: getVar('--ki-grey', '#A08898'),
        titleColor: getVar('--ki-dark-plum', '#F5EDF3'),
        annotationColor: getVar('--ki-grey', '#A08898'),

        // Status
        error: getVar('--status-error-text', '#EF9A9A'),
        success: getVar('--status-success-text', '#A5D6A7')
    };
}

// Returns a theme-aware Plotly layout base (merge with your specific layout)
function getPlotlyThemeLayout() {
    var c = getThemeColors();
    return {
        paper_bgcolor: c.paperBg,
        plot_bgcolor: c.plotBg,
        font: { color: c.text, family: 'Roboto, sans-serif' },
        xaxis: {
            color: c.axisColor,
            gridcolor: c.gridColor,
            zerolinecolor: c.gridColor
        },
        yaxis: {
            color: c.axisColor,
            gridcolor: c.gridColor,
            zerolinecolor: c.gridColor
        }
    };
}

document.addEventListener('DOMContentLoaded', function() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    updateThemeIcon(theme);
});
