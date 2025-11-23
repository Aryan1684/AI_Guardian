// ============================================
// AI DETECTION SYSTEM - MAIN.JS
// Destiny Coder's 2025 - Complete Clean Version
// ============================================

const state = {
    currentTab: 'universal',
    currentUploadType: null,
    isProcessing: false
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('%c🔥 AI Detection System Loaded', 'color: #6366f1; font-size: 20px; font-weight: bold;');
    console.log('%cDestiny Coder\'s 2025', 'color: #ec4899; font-size: 14px;');
    
    initializeTabs();
    initializeContentTypeCards();
    initializeFileUploads();
    initializeTextInput();
    initializeInputToggle();
    initializeDragAndDrop();
    initializeEmailInput();
    
    console.log('✅ All systems ready!');
});

// ============================================
// TAB NAVIGATION
// ============================================

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Update active panel
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    const activePanel = document.getElementById(tabName);
    if (activePanel) {
        activePanel.classList.add('active');
    }
    
    hideUploadSection();
    state.currentTab = tabName;
    
    console.log(`📑 Tab switched to: ${tabName}`);
}

// ============================================
// CONTENT TYPE CARDS
// ============================================

function initializeContentTypeCards() {
    const typeCards = document.querySelectorAll('.type-card');
    
    typeCards.forEach(card => {
        card.addEventListener('click', () => {
            const type = card.dataset.type;
            showUploadSection(type);
        });
    });
}

function showUploadSection(type) {
    // Hide all upload sections
    document.querySelectorAll('.upload-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Hide content type grid
    const grid = document.querySelector('.content-type-grid');
    if (grid) {
        grid.style.display = 'none';
    }
    
    // Show selected section
    const section = document.getElementById(`${type}-section`);
    if (section) {
        section.style.display = 'block';
    }
    
    state.currentUploadType = type;
    console.log(`📂 Opened ${type} section`);
}

function hideUploadSection() {
    // Hide all upload sections
    document.querySelectorAll('.upload-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show content type grid if on universal tab
    if (state.currentTab === 'universal') {
        const grid = document.querySelector('.content-type-grid');
        if (grid) {
            grid.style.display = 'grid';
        }
    }
    
    // Clear results and previews
    document.querySelectorAll('.result-panel').forEach(panel => {
        panel.innerHTML = '';
        panel.style.display = 'none';
    });
    
    document.querySelectorAll('.dropzone-preview').forEach(preview => {
        preview.innerHTML = '';
        preview.style.display = 'none';
    });
    
    state.currentUploadType = null;
}

// ============================================
// FILE UPLOAD HANDLING
// ============================================

function initializeFileUploads() {
    const fileInputs = [
        { id: 'video-input', type: 'video' },
        { id: 'image-input', type: 'image' },
        { id: 'audio-input', type: 'audio' },
        { id: 'deepfake-input', type: 'deepfake' }
    ];
    
    fileInputs.forEach(({ id, type }) => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('change', (e) => handleFileSelect(e, type));
        }
    });
}

function handleFileSelect(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    console.log(`📎 File selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    
    if (!validateFile(file, type)) {
        return;
    }
    
    showFilePreview(file, type);
    processFile(file, type);
}

function validateFile(file, type) {
    const validations = {
        video: {
            types: ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'],
            maxSize: 100 * 1024 * 1024
        },
        image: {
            types: ['image/jpeg', 'image/png', 'image/webp', 'image/gif'],
            maxSize: 20 * 1024 * 1024
        },
        audio: {
            types: ['audio/mpeg', 'audio/wav', 'audio/mp4'],
            maxSize: 50 * 1024 * 1024
        },
        deepfake: {
            types: ['video/mp4', 'image/jpeg', 'image/png'],
            maxSize: 100 * 1024 * 1024
        }
    };
    
    const v = validations[type];
    
    if (!v.types.includes(file.type)) {
        showToast('error', 'Invalid File', 'Please upload a supported format.');
        return false;
    }
    
    if (file.size > v.maxSize) {
        showToast('error', 'File Too Large', `Max: ${v.maxSize / 1024 / 1024}MB`);
        return false;
    }
    
    return true;
}

function showFilePreview(file, type) {
    const preview = document.getElementById(`${type}-preview`);
    if (!preview) return;
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
        let html = '<div style="margin-top: 1rem;">';
        
        if (file.type.startsWith('video')) {
            html += `<video src="${e.target.result}" controls style="max-width: 100%; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></video>`;
        } else if (file.type.startsWith('image')) {
            html += `<img src="${e.target.result}" style="max-width: 100%; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">`;
        } else if (file.type.startsWith('audio')) {
            html += `<audio src="${e.target.result}" controls style="width: 100%;"></audio>`;
        }
        
        html += `<p style="text-align: center; margin-top: 1rem; color: #94a3b8;">${file.name}</p>`;
        html += '</div>';
        
        preview.innerHTML = html;
        preview.style.display = 'block';
    };
    
    reader.readAsDataURL(file);
}

// ============================================
// DRAG AND DROP
// ============================================

function initializeDragAndDrop() {
    const dropzones = document.querySelectorAll('.premium-dropzone');
    
    dropzones.forEach(dropzone => {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const dropzoneId = dropzone.id;
                const type = dropzoneId.replace('-dropzone', '');
                const input = document.getElementById(`${type}-input`);
                
                if (input) {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(files[0]);
                    input.files = dataTransfer.files;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });
    });
}

// ============================================
// TEXT INPUT
// ============================================

function initializeTextInput() {
    const textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.addEventListener('input', function() {
            updateTextStats(this, 'char-count', 'word-count');
        });
    }
    
    const newsText = document.getElementById('news-text');
    if (newsText) {
        newsText.addEventListener('input', function() {
            updateTextStats(this, 'news-char-count', 'news-word-count');
        });
    }
}

function updateTextStats(textarea, charId, wordId) {
    const text = textarea.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    
    const charEl = document.getElementById(charId);
    const wordEl = document.getElementById(wordId);
    
    if (charEl) charEl.textContent = `${chars} characters`;
    if (wordEl) wordEl.textContent = `${words} words`;
}

function analyzeText() {
    const input = document.getElementById('text-input');
    if (!input) {
        showToast('error', 'Error', 'Text input not found');
        return;
    }
    
    const text = input.value.trim();
    
    if (!text) {
        showToast('warning', 'No Text', 'Please enter text to analyze');
        input.focus();
        return;
    }
    
    if (text.length < 50) {
        showToast('warning', 'Too Short', 'Enter at least 50 characters');
        input.focus();
        return;
    }
    
    processText(text, 'text');
}

// ============================================
// INPUT TOGGLE
// ============================================

function initializeInputToggle() {
    const toggleButtons = document.querySelectorAll('.toggle-btn');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const inputType = button.dataset.input;
            
            // Update active button
            toggleButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show/hide containers
            document.querySelectorAll('.input-container').forEach(container => {
                container.style.display = 'none';
            });
            
            const target = document.getElementById(`news-${inputType}-input`);
            if (target) {
                target.style.display = 'block';
            }
            
            console.log(`🔄 Switched to ${inputType} input`);
        });
    });
}

function checkNews() {
    console.log('🔍 checkNews() called');
    
    const activeBtn = document.querySelector('.toggle-btn.active');
    if (!activeBtn) {
        showToast('error', 'Error', 'No input type selected');
        return;
    }
    
    const inputType = activeBtn.dataset.input;
    console.log(`Input type: ${inputType}`);
    
    if (inputType === 'text') {
        const newsText = document.getElementById('news-text');
        if (!newsText) {
            console.error('news-text element not found');
            showToast('error', 'Error', 'Text area not found');
            return;
        }
        
        const text = newsText.value.trim();
        console.log(`Text length: ${text.length}`);
        
        if (!text) {
            showToast('warning', 'No Text', 'Enter news text to check');
            newsText.focus();
            return;
        }
        
        if (text.length < 50) {
            showToast('warning', 'Too Short', 'Enter at least 50 characters');
            newsText.focus();
            return;
        }
        
        console.log('✅ Processing news text...');
        processText(text, 'news');
        
    } else if (inputType === 'url') {
        const newsUrl = document.getElementById('news-url');
        if (!newsUrl) {
            showToast('error', 'Error', 'URL input not found');
            return;
        }
        
        const url = newsUrl.value.trim();
        
        if (!url) {
            showToast('warning', 'No URL', 'Enter a URL to check');
            newsUrl.focus();
            return;
        }
        
        if (!url.startsWith('http')) {
            showToast('error', 'Invalid URL', 'URL must start with http:// or https://');
            newsUrl.focus();
            return;
        }
        
        processURL(url);
    }
}

// ============================================
// EMAIL SUBSCRIPTION
// ============================================

function initializeEmailInput() {
    const emailInput = document.getElementById('notify-email');
    if (emailInput) {
        emailInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                subscribeEmail();
            }
        });
    }
}

async function subscribeEmail() {
    const emailInput = document.getElementById('notify-email');
    const email = emailInput.value.trim();
    const messageDiv = document.getElementById('subscribe-message');
    const button = document.querySelector('.btn-notify');
    
    // Clear previous message
    if (messageDiv) {
        messageDiv.className = 'subscribe-message';
        messageDiv.textContent = '';
        messageDiv.style.display = 'none';
    }
    
    // Validate email
    if (!email) {
        showSubscribeMessage('error', '⚠️ Enter your email address');
        emailInput.focus();
        return;
    }
    
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) {
        showSubscribeMessage('error', '⚠️ Enter a valid email');
        emailInput.focus();
        return;
    }
    
    // Disable button
    const originalHTML = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Subscribing...';
    
    try {
        const response = await fetch('/api/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSubscribeMessage('success', result.message);
            emailInput.value = '';
            createConfetti();
            showToast('success', 'Subscribed! 🎉', result.message);
        } else {
            showSubscribeMessage('error', result.error);
        }
    } catch (error) {
        showSubscribeMessage('error', '❌ Failed to subscribe');
        console.error(error);
    } finally {
        button.disabled = false;
        button.innerHTML = originalHTML;
    }
}

function showSubscribeMessage(type, message) {
    const div = document.getElementById('subscribe-message');
    if (div) {
        div.className = `subscribe-message ${type}`;
        div.textContent = message;
        div.style.display = 'block';
    }
}

function createConfetti() {
    const colors = ['#6366f1', '#ec4899', '#10b981', '#f59e0b'];
    for (let i = 0; i < 50; i++) {
        setTimeout(() => {
            const c = document.createElement('div');
            Object.assign(c.style, {
                position: 'fixed',
                width: '10px',
                height: '10px',
                backgroundColor: colors[Math.floor(Math.random() * colors.length)],
                left: Math.random() * 100 + '%',
                top: '-20px',
                borderRadius: Math.random() > 0.5 ? '50%' : '0',
                zIndex: '10001',
                pointerEvents: 'none'
            });
            document.body.appendChild(c);
            c.animate([
                { transform: 'translateY(0)', opacity: 1 },
                { transform: `translateY(${window.innerHeight}px)`, opacity: 0 }
            ], { duration: 3000 }).onfinish = () => c.remove();
        }, i * 20);
    }
}

// ============================================
// PROCESSING FUNCTIONS
// ============================================

async function processFile(file, type) {
    if (state.isProcessing) {
        showToast('warning', 'Busy', 'Please wait...');
        return;
    }
    
    state.isProcessing = true;
    showLoading(`Analyzing ${type}...`, 'Processing file...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        console.log(`⏳ Uploading ${type}...`);
        simulateProgress();
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        hideLoading();
        displayResult(result, type);
        showToast('success', 'Complete', 'Analysis finished!');
        
        console.log('✅ Analysis complete:', result);
        
    } catch (error) {
        hideLoading();
        showToast('error', 'Failed', error.message);
        console.error('❌ Error:', error);
    } finally {
        state.isProcessing = false;
    }
}

async function processText(text, type) {
    if (state.isProcessing) {
        showToast('warning', 'Busy', 'Please wait...');
        return;
    }
    
    state.isProcessing = true;
    showLoading('Analyzing text...', 'Processing content...');
    
    try {
        console.log(`⏳ Analyzing ${type} (${text.length} chars)...`);
        simulateProgress();
        
        const response = await fetch('/api/analyze-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, type })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Result received:', result);
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        hideLoading();
        displayResult(result, type);
        showToast('success', 'Complete', 'Analysis finished!');
        
        console.log('✅ Text analysis complete');
        
    } catch (error) {
        hideLoading();
        showToast('error', 'Failed', error.message);
        console.error('❌ Error:', error);
    } finally {
        state.isProcessing = false;
    }
}

async function processURL(url) {
    if (state.isProcessing) {
        showToast('warning', 'Busy', 'Please wait...');
        return;
    }
    
    state.isProcessing = true;
    showLoading('Fetching article...', 'Analyzing URL...');
    
    try {
        simulateProgress();
        
        const response = await fetch('/api/analyze-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        hideLoading();
        displayResult(result, 'news');
        showToast('success', 'Complete', 'URL analyzed!');
        
    } catch (error) {
        hideLoading();
        showToast('error', 'Failed', error.message);
        console.error(error);
    } finally {
        state.isProcessing = false;
    }
}

// ============================================
// DISPLAY RESULT
// ============================================

function displayResult(result, type) {
    console.log(`📊 Displaying result for ${type}:`, result);
    
    const panel = document.getElementById(`${type}-result`);
    if (!panel) {
        console.error(`❌ Panel not found: ${type}-result`);
        showToast('error', 'Error', 'Result panel not found');
        return;
    }
    
    const score = result.ai_probability || result.score || 0.5;
    const scoreClass = score >= 0.7 ? 'danger' : (score >= 0.4 ? 'warning' : 'success');
    const scoreEmoji = score >= 0.7 ? '🚨' : (score >= 0.4 ? '⚠️' : '✅');
    
    panel.innerHTML = `
        <div class="result-header">
            <h3 class="result-title">${scoreEmoji} Analysis Result</h3>
            <div class="result-score">
                <span class="score-label">${type === 'news' ? 'Fake News' : 'AI'} Probability</span>
                <span class="score-value ${scoreClass}">${Math.round(score * 100)}%</span>
            </div>
        </div>
        
        <div class="result-details">
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-icon"><i class="fas fa-brain"></i></div>
                    <div class="detail-content">
                        <span class="detail-label">Classification</span>
                        <span class="detail-value">${result.classification || 'Unknown'}</span>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-icon"><i class="fas fa-gauge-high"></i></div>
                    <div class="detail-content">
                        <span class="detail-label">Confidence</span>
                        <span class="detail-value">${Math.round((result.confidence || 0.75) * 100)}%</span>
                    </div>
                </div>
            </div>
            
            ${result.details ? `
            <div class="result-explanation">
                <h4><i class="fas fa-info-circle"></i> Details</h4>
                <p>${result.details}</p>
            </div>
            ` : ''}
            
            ${result.specific_findings && result.specific_findings.length > 0 ? `
            <div class="result-recommendations">
                <h4><i class="fas fa-search"></i> Findings</h4>
                <ul>${result.specific_findings.map(f => `<li>${f}</li>`).join('')}</ul>
            </div>
            ` : ''}
            
            ${result.recommendations && result.recommendations.length > 0 ? `
            <div class="result-recommendations">
                <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
                <ul>${result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
            </div>
            ` : ''}
        </div>
    `;
    
    // Show panel
    panel.style.display = 'block';
    panel.style.visibility = 'visible';
    panel.style.opacity = '1';
    
    // Scroll to result
    setTimeout(() => {
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
    console.log('✅ Result displayed successfully');
}

// ============================================
// LOADING
// ============================================

function showLoading(message, submessage) {
    const overlay = document.getElementById('loading-overlay');
    const msgEl = document.getElementById('loading-message');
    const subEl = document.getElementById('loading-submessage');
    
    if (msgEl) msgEl.textContent = message;
    if (subEl) subEl.textContent = submessage;
    if (overlay) overlay.classList.add('active');
    
    console.log(`⏳ ${message}`);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.remove('active');
    
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    if (fill) fill.style.width = '0%';
    if (text) text.textContent = '0%';
}

function simulateProgress() {
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15 + 5;
        if (progress >= 95) {
            progress = 95;
            clearInterval(interval);
        }
        if (fill) fill.style.width = `${progress}%`;
        if (text) text.textContent = `${Math.round(progress)}%`;
    }, 200);
}

// ============================================
// TOAST NOTIFICATIONS
// ============================================

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-icon">${icons[type]}</div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
    
    console.log(`📢 ${type.toUpperCase()}: ${title} - ${message}`);
}

// ============================================
// EXPOSE GLOBAL FUNCTIONS
// ============================================

window.hideUploadSection = hideUploadSection;
window.analyzeText = analyzeText;
window.checkNews = checkNews;
window.subscribeEmail = subscribeEmail;

console.log('✅ main.js loaded successfully');