/**
 * Unified Navigation for All Pages
 * Ensures consistent dropdown navigation across the website
 */

function createNavigation() {
    return `
        <nav class="navbar">
            <div class="nav-container">
                <div class="nav-brand">
                    <span class="brand-logo">trustcv</span>
                    <span class="brand-tagline">Trustworthy Cross-Validation Toolkit</span>
                </div>
                <button class="nav-toggle" onclick="toggleMobileMenu()">
                    <span class="material-icons">menu</span>
                </button>
                <ul class="nav-menu" id="navMenu">
                    <li><a href="index.html">Home</a></li>
                    <li class="nav-dropdown">
                        <a href="#">Learn</a>
                        <div class="dropdown-menu">
                            <a href="demonstrations.html">
                                <span class="material-icons" style="font-size: 18px;">play_circle</span>
                                Demonstrations
                            </a>
                            <a href="quiz.html">
                                <span class="material-icons" style="font-size: 18px;">quiz</span>
                                Interactive Quiz
                            </a>
                            <a href="notebooks.html">
                                <span class="material-icons" style="font-size: 18px;">menu_book</span>
                                Notebooks
                            </a>
                        </div>
                    </li>
                    <li class="nav-dropdown">
                        <a href="#">Resources</a>
                        <div class="dropdown-menu">
                            <a href="methods-comparison.html">
                                <span class="material-icons" style="font-size: 18px;">table_chart</span>
                                Methods Guide
                            </a>
                            <a href="docs/API_REFERENCE.html">
                                <span class="material-icons" style="font-size: 18px;">api</span>
                                API Documentation
                            </a>
                            <div class="dropdown-divider"></div>
                            <a href="docs/CV_SELECTION_GUIDE.md">
                                <span class="material-icons" style="font-size: 18px;">help</span>
                                Selection Guide
                            </a>
                        </div>
                    </li>
                    <li class="nav-dropdown">
                        <a href="#">Regulatory</a>
                        <div class="dropdown-menu">
                            <a href="regulatory-cv-tutorial.html">
                                <span class="material-icons" style="font-size: 18px;">school</span>
                                Guidelines Tutorial
                            </a>
                            <a href="regulatory-report.html">
                                <span class="material-icons" style="font-size: 18px;">description</span>
                                Report Generator
                            </a>
                            <div class="dropdown-divider"></div>
                            <a href="docs/REGULATORY_CV_GUIDELINES.md">
                                <span class="material-icons" style="font-size: 18px;">article</span>
                                Documentation
                            </a>
                        </div>
                    </li>
                    <li><a href="https://github.com/ki-smile/trustcv" class="nav-github">
                        <span class="material-icons">code</span> GitHub
                    </a></li>
                </ul>
            </div>
        </nav>
    `;
}

// Mobile menu toggle function
function toggleMobileMenu() {
    const menu = document.getElementById('navMenu');
    menu.classList.toggle('active');
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    const menu = document.getElementById('navMenu');
    const toggle = document.querySelector('.nav-toggle');
    
    if (menu && toggle && !menu.contains(event.target) && !toggle.contains(event.target)) {
        menu.classList.remove('active');
    }
});

// Function to inject navigation into page
function injectNavigation() {
    const navPlaceholder = document.getElementById('navigation-placeholder');
    if (navPlaceholder) {
        navPlaceholder.innerHTML = createNavigation();
    }
}

// Auto-inject on DOM ready if placeholder exists
document.addEventListener('DOMContentLoaded', injectNavigation);