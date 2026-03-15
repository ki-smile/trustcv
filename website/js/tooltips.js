/**
 * Interactive Tooltips for Technical Terms
 */

const tooltipsData = {
    'Data Leakage': 'When information from outside the training dataset is used to create the model, leading to overoptimistic performance.',
    'Look-ahead bias': 'A type of data leakage where future information is used to predict past or current events.',
    'Purged K-Fold': 'A cross-validation method that removes samples near the training/test split to prevent temporal leakage.',
    'Embargo': 'A technique that removes samples following the test set to prevent information leakage into subsequent training sets.',
    'I.I.D.': 'Independent and Identically Distributed - the assumption that samples are independent and drawn from the same distribution.',
    'Stratified': 'Ensuring that each fold maintains the same proportion of target classes as the complete dataset.',
    'Temporal Leakage': 'Leakage occurring through time-dependent relationships in the data.',
    'Spatial Block': 'Dividing data into geographic regions to test model generalization across different locations.',
    'GMLP': 'Good Machine Learning Practice - a set of guiding principles for medical AI development.',
    'TRIPOD+AI': 'Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis - specific to AI.',
    '510(k)': 'A premarket submission made to the FDA to demonstrate that a medical device is safe and effective.',
    'CE MDR': 'European Medical Device Regulation - the regulatory framework for medical devices in the EU.'
};

function initTooltips() {
    // Find all text elements that might contain these terms
    const elements = document.querySelectorAll('p, li, td, .demo-description, .notebook-description');
    
    elements.forEach(el => {
        let html = el.innerHTML;
        let modified = false;
        
        // We only want to wrap terms that aren't already wrapped in a tooltip or link
        if (el.closest('.tooltip-trigger') || el.closest('a') || el.closest('button')) return;

        Object.keys(tooltipsData).forEach(term => {
            // Regex to match term (case insensitive, whole word)
            const regex = new RegExp(`\\b(${term})\\b`, 'gi');
            
            // Only replace if not already part of an HTML tag
            if (html.includes(`>${term}<`) || html.includes(`"${term}"`)) return;

            if (regex.test(html)) {
                html = html.replace(regex, `<span class="tooltip-trigger">$1<span class="tooltip-content">${tooltipsData[term]}</span></span>`);
                modified = true;
            }
        });
        
        if (modified) {
            el.innerHTML = html;
        }
    });
}

document.addEventListener('DOMContentLoaded', initTooltips);
