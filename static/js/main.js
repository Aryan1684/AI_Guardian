// ============================================
// AI GUARDIAN - OPTIMIZED CORE
// ============================================

const state = {
    currentTab: 'universal',
    mode: 'text' // for news tab
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('⚡ AI Guardian System Online');
    initTabs();
    initUniversalSelector();
    initDropzones();
});

// ============================================
// TAB LOGIC
// ============================================

function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            // UI Update
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active'); // Instant feedback

            // Panel Update
            const target = btn.dataset.tab;
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(target).classList.add('active');

            state.currentTab = target;
        });
    });
}

// ============================================
// UNIVERSAL TAB SELECTOR
// ============================================

function initUniversalSelector() {
    const cards = document.querySelectorAll('.selection-card');
    const selector = document.getElementById('type-selector');
    const uploadArea = document.getElementById('universal-upload-area');
    const title = document.getElementById('upload-title');
    const dropzone = document.getElementById('universal-dropzone');
    const textArea = document.getElementById('universal-text-area');

    cards.forEach(card => {
        card.addEventListener('click', () => {
            const type = card.dataset.type;

            // Switch View
            selector.style.display = 'none';
            uploadArea.style.display = 'block';
            title.textContent = `${type.charAt(0).toUpperCase() + type.slice(1)} Content Analysis`;

            // Reset Contents
            document.getElementById('universal-result').style.display = 'none';
            document.getElementById('universal-preview').innerHTML = '';

            // Toggle Input Type
            if (type === 'text') {
                dropzone.style.display = 'none';
                textArea.style.display = 'block';
            } else {
                dropzone.style.display = 'block';
                textArea.style.display = 'none';
                // Update Accept Attribute
                const input = document.getElementById('universal-file-input');
                if (type === 'video') input.accept = 'video/*';
                if (type === 'image') input.accept = 'image/*';
                if (type === 'audio') input.accept = 'audio/*';
            }
        });
    });

    // File Input Trigger
    document.getElementById('universal-dropzone').addEventListener('click', () => {
        document.getElementById('universal-file-input').click();
    });

    // File Input Change
    document.getElementById('universal-file-input').addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0], 'universal');
    });
}

function resetUniversalTab() {
    document.getElementById('type-selector').style.display = 'grid';
    document.getElementById('universal-upload-area').style.display = 'none';
}

// ============================================
// DROPZONE LOGIC
// ============================================

function initDropzones() {
    const zones = document.querySelectorAll('.dropzone');

    zones.forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) {
                // Determine source/type based on zone ID
                const type = zone.id.includes('universal') ? 'universal' : 'deepfake';
                handleFile(files[0], type);
            }
        });
    });

    // Deepfake specific input
    const dfInput = document.getElementById('deepfake-file-input');
    if (dfInput) dfInput.addEventListener('change', (e) => handleFile(e.target.files[0], 'deepfake'));
}


// ============================================
// CORE PROCESSING (OPTIMIZED)
// ============================================

async function handleFile(file, context) {
    if (!file) return;

    // Fast Preview
    const previewId = context === 'universal' ? 'universal-preview' : 'deepfake-preview';
    const previewEl = document.getElementById(previewId);
    previewEl.innerHTML = `<p class="upload-hint">Selected: ${file.name}</p>`;

    // Prepare Request
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', file.type.split('/')[0]); // simplified type detection

    // Show simplified loading (NON-BLOCKING)
    const resultId = context === 'universal' ? 'universal-result' : 'deepfake-result';
    const resultEl = document.getElementById(resultId);
    resultEl.style.display = 'block';
    resultEl.innerHTML = `<div style="text-align:center; padding: 2rem; color: #10b981;"><i class="fas fa-circle-notch fa-spin fa-2x"></i><p>Analyzing...</p></div>`;

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        renderResult(data, resultEl);
    } catch (e) {
        resultEl.innerHTML = `<p style="color:var(--danger)">Error: ${e.message}</p>`;
    }
}

async function analyzeText() {
    const text = document.getElementById('text-analysis-input').value;
    if (text.length < 10) return alert('Please enter at least 10 characters');

    const resultEl = document.getElementById('universal-result');
    resultEl.style.display = 'block';
    resultEl.innerHTML = `<div style="text-align:center; padding:1rem;"><i class="fas fa-circle-notch fa-spin"></i> Processing...</div>`;

    try {
        const res = await fetch('/api/analyze-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, type: 'text' })
        });
        const data = await res.json();
        renderResult(data, resultEl);
    } catch (e) {
        resultEl.innerHTML = `<p style="color:var(--danger)">High Load: Please try again</p>`;
    }
}

// ============================================
// NEWS TAB LOGIC
// ============================================

function setNewsMode(mode) {
    state.mode = mode;
    document.querySelectorAll('.toggle-pill').forEach(b => b.classList.toggle('active'));

    if (mode === 'text') {
        document.getElementById('news-text-mode').style.display = 'block';
        document.getElementById('news-url-mode').style.display = 'none';
    } else {
        document.getElementById('news-text-mode').style.display = 'none';
        document.getElementById('news-url-mode').style.display = 'block';
    }
}

async function analyzeNews() {
    const resultEl = document.getElementById('news-result');
    resultEl.style.display = 'block';
    resultEl.innerHTML = `<div style="text-align:center; color: var(--primary)">Processing...</div>`;

    // Simulate Request for Demo (To match old functionality but faster)
    try {
        // Build payload based on mode
        let payload = {};
        let endpoint = '/api/analyze-text';

        if (state.mode === 'text') {
            payload = { text: document.getElementById('news-input-text').value, type: 'news' };
        } else {
            endpoint = '/api/analyze-url';
            payload = { url: document.getElementById('news-input-url').value };
        }

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderResult(data, resultEl);

    } catch (e) {
        resultEl.innerHTML = `<p>Error connecting to analysis engine.</p>`;
    }
}

// ============================================
// RENDERER
// ============================================

function renderResult(data, container) {
    const isAi = data.ai_probability > 0.5;
    const badgeClass = isAi ? 'danger' : 'success';
    const percent = Math.round((data.ai_probability || 0) * 100);

    const html = `
        <div class="result-card">
            <div class="result-header">
                <h3>Analysis Complete</h3>
                <span class="score-badge ${badgeClass}">${isAi ? 'AI / Fake' : 'Human / Real'} ${percent}%</span>
            </div>
            <div class="detail-row">
                <span>Confidence</span>
                <span>${(data.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="detail-row">
                <span>Key Findings</span>
                <span style="color: var(--text-muted)">${data.specific_findings?.[0] || 'Standard pattern'}</span>
            </div>
            <p style="margin-top:1rem; font-size:0.9rem; color:var(--text-muted)">${data.details}</p>
        </div>
    `;

    container.innerHTML = html;
}

// ============================================
// PROCTORING / SUBSCRIPTION
// ============================================

function subscribe() {
    const email = document.getElementById('notify-email').value;
    if (email) {
        alert('Subscribed! We will notify you.');
        document.getElementById('notify-email').value = '';
    }
}