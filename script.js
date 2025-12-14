/**
 * Cyberbullying Detection System - Main JavaScript
 * Handles detection form submission, API calls, and UI interactions
 * 
 * Author: College Project Team
 * Date: December 2025
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDetectionForm();
    initializeCharacterCounter();
    initializeExamples();
    initializeSmoothScroll();
});

/**
 * Initialize the detection form functionality
 */
function initializeDetectionForm() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeAgainBtn = document.getElementById('analyzeAgainBtn');
    const textInput = document.getElementById('textInput');
    
    // Only initialize if we're on the detection page
    if (!analyzeBtn) return;
    
    // Analyze button click handler
    analyzeBtn.addEventListener('click', function() {
        const text = textInput.value.trim();
        
        if (!text) {
            showAlert('Please enter some text to analyze.');
            textInput.focus();
            return;
        }
        
        analyzeText(text);
    });
    
    // Clear button click handler
    clearBtn.addEventListener('click', function() {
        textInput.value = '';
        updateCharacterCount();
        hideResult();
        textInput.focus();
    });
    
    // Analyze again button click handler
    if (analyzeAgainBtn) {
        analyzeAgainBtn.addEventListener('click', function() {
            hideResult();
            textInput.value = '';
            updateCharacterCount();
            textInput.focus();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
    
    // Enter key to submit (Ctrl+Enter for textarea)
    textInput.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });
}

/**
 * Initialize character counter for textarea
 */
function initializeCharacterCounter() {
    const textInput = document.getElementById('textInput');
    
    if (!textInput) return;
    
    textInput.addEventListener('input', updateCharacterCount);
}

/**
 * Update character count display
 */
function updateCharacterCount() {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    
    if (textInput && charCount) {
        charCount.textContent = textInput.value.length;
    }
}

/**
 * Initialize example cards click functionality
 */
function initializeExamples() {
    // Example click handler is defined globally as setExample()
}

/**
 * Set example text in the textarea
 * @param {HTMLElement} element - The clicked example card element
 */
function setExample(element) {
    const textInput = document.getElementById('textInput');
    const exampleText = element.querySelector('p').textContent;
    
    if (textInput) {
        textInput.value = exampleText;
        updateCharacterCount();
        textInput.focus();
        
        // Scroll to the form
        textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

/**
 * Analyze the text by sending it to the backend API
 * @param {string} text - The text to analyze
 */
async function analyzeText(text) {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const detectionForm = document.querySelector('.detection-form');
    const resultSection = document.getElementById('resultSection');
    
    // Show loading state
    detectionForm.style.display = 'none';
    loadingIndicator.style.display = 'block';
    resultSection.style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            throw new Error(data.message || 'Failed to analyze text');
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error analyzing text: ' + error.message);
        hideResult();
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

/**
 * Display the analysis result
 * @param {Object} data - The result data from the API
 */
function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultLabel = document.getElementById('resultLabel');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const resultMessage = document.getElementById('resultMessage');
    const analyzedText = document.getElementById('analyzedText');
    
    // Remove previous classes
    resultCard.classList.remove('bullying', 'non-bullying', 'neutral');
    
    // Set result based on prediction label
    let icon, cardClass;
    
    switch (data.label) {
        case 'bullying':
            icon = '❌';
            cardClass = 'bullying';
            break;
        case 'non-bullying':
            icon = '✅';
            cardClass = 'non-bullying';
            break;
        default:
            icon = '⚠️';
            cardClass = 'neutral';
    }
    
    // Update UI elements
    resultCard.classList.add(cardClass);
    resultIcon.textContent = icon;
    resultLabel.textContent = data.prediction;
    
    // Animate confidence bar
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceValue.textContent = confidencePercent + '%';
    
    // Delay for animation effect
    setTimeout(() => {
        confidenceFill.style.width = confidencePercent + '%';
    }, 100);
    
    resultMessage.textContent = data.message;
    analyzedText.textContent = '"' + data.original_text + '"';
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Hide the result section and show the form
 */
function hideResult() {
    const detectionForm = document.querySelector('.detection-form');
    const resultSection = document.getElementById('resultSection');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const confidenceFill = document.getElementById('confidenceFill');
    
    if (detectionForm) detectionForm.style.display = 'block';
    if (resultSection) resultSection.style.display = 'none';
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    if (confidenceFill) confidenceFill.style.width = '0%';
}

/**
 * Show an alert message to the user
 * @param {string} message - The message to display
 */
function showAlert(message) {
    // Create custom alert if it doesn't exist
    let alertBox = document.getElementById('customAlert');
    
    if (!alertBox) {
        alertBox = document.createElement('div');
        alertBox.id = 'customAlert';
        alertBox.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            z-index: 9999;
            animation: slideIn 0.3s ease;
        `;
        document.body.appendChild(alertBox);
    }
    
    alertBox.textContent = message;
    alertBox.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        alertBox.style.display = 'none';
    }, 3000);
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/**
 * Add slide-in animation keyframes dynamically
 */
(function addAnimations() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }
    `;
    document.head.appendChild(style);
})();

/**
 * Mobile navigation toggle (if needed in future)
 */
function toggleMobileNav() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('active');
}

// Expose setExample to global scope for onclick handlers
window.setExample = setExample;
