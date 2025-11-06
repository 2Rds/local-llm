// API Base URL
const API_BASE = 'http://localhost:5000/api';

// State
let currentModel = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeSearch();
    initializeChat();
    loadDownloadedModels();
    updateModelStatus();

    // Auto-refresh status every 5 seconds
    setInterval(updateModelStatus, 5000);
});

// Tab management
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');

            // Refresh models when switching to my-models tab
            if (tabName === 'my-models') {
                loadDownloadedModels();
            }
        });
    });
}

// Search functionality
function initializeSearch() {
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');
    const suggestions = document.querySelectorAll('.model-suggestion');

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    suggestions.forEach(suggestion => {
        suggestion.addEventListener('click', () => {
            searchInput.value = suggestion.dataset.model;
            performSearch();
        });
    });

    // Load popular models on start
    performSearch();
}

async function performSearch() {
    const query = document.getElementById('search-input').value;
    const resultsDiv = document.getElementById('search-results');
    const loadingDiv = document.getElementById('search-loading');

    resultsDiv.innerHTML = '';
    loadingDiv.style.display = 'block';

    try {
        const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&limit=20`);
        const data = await response.json();

        loadingDiv.style.display = 'none';

        if (data.success && data.models.length > 0) {
            data.models.forEach(model => {
                const card = createModelCard(model, false);
                resultsDiv.appendChild(card);
            });
        } else {
            resultsDiv.innerHTML = '<div class="empty-state"><p>No models found. Try a different search term.</p></div>';
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        showToast('Error searching models: ' + error.message, 'error');
    }
}

// Create model card
function createModelCard(model, isDownloaded) {
    const card = document.createElement('div');
    card.className = 'model-card';

    const tagsHtml = model.tags && model.tags.length > 0
        ? `<div class="model-tags">${model.tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}</div>`
        : '';

    const sizeHtml = model.size_mb
        ? `<div class="model-size">${model.size_mb} MB</div>`
        : '';

    const stats = !isDownloaded
        ? `<div class="model-stats">
            <span>⬇️ ${formatNumber(model.downloads)}</span>
            <span>❤️ ${formatNumber(model.likes)}</span>
           </div>`
        : '';

    const actions = isDownloaded
        ? `<div class="model-actions">
            <button class="btn btn-success" onclick="loadModel('${model.id}')">Load</button>
            <button class="btn btn-danger" onclick="deleteModel('${model.id}')">Delete</button>
           </div>`
        : `<div class="model-actions">
            <button class="btn btn-primary" onclick="downloadModel('${model.id}')">Download</button>
           </div>`;

    card.innerHTML = `
        <h3>${model.name}</h3>
        <div class="model-author">by ${model.author}</div>
        ${sizeHtml}
        ${stats}
        ${tagsHtml}
        ${actions}
    `;

    return card;
}

// Download model
async function downloadModel(modelId) {
    showToast(`Downloading ${modelId}... This may take several minutes.`, 'info');

    try {
        const response = await fetch(`${API_BASE}/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Successfully downloaded ${modelId}`, 'success');
            loadDownloadedModels();
        } else {
            showToast(`Error downloading model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error downloading model: ${error.message}`, 'error');
    }
}

// Load downloaded models
async function loadDownloadedModels() {
    const modelsDiv = document.getElementById('downloaded-models');
    const loadingDiv = document.getElementById('models-loading');
    const noModelsDiv = document.getElementById('no-models');
    const refreshBtn = document.getElementById('refresh-models-btn');

    modelsDiv.innerHTML = '';
    loadingDiv.style.display = 'block';
    noModelsDiv.style.display = 'none';

    refreshBtn.addEventListener('click', loadDownloadedModels);

    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();

        loadingDiv.style.display = 'none';

        if (data.success && data.models.length > 0) {
            data.models.forEach(model => {
                const card = createModelCard(model, true);
                modelsDiv.appendChild(card);
            });
        } else {
            noModelsDiv.style.display = 'block';
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        showToast('Error loading models: ' + error.message, 'error');
    }
}

// Delete model
async function deleteModel(modelId) {
    if (!confirm(`Are you sure you want to delete ${modelId}?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/models/${modelId.replace('/', '--')}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Successfully deleted ${modelId}`, 'success');
            loadDownloadedModels();
            updateModelStatus();
        } else {
            showToast(`Error deleting model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error deleting model: ${error.message}`, 'error');
    }
}

// Load model for inference
async function loadModel(modelId) {
    showToast(`Loading ${modelId}...`, 'info');

    try {
        const response = await fetch(`${API_BASE}/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });

        const data = await response.json();

        if (data.success) {
            currentModel = modelId;
            showToast(`Successfully loaded ${modelId} on ${data.device}`, 'success');
            updateModelStatus();

            // Switch to chat tab
            document.querySelector('[data-tab="chat"]').click();
        } else {
            showToast(`Error loading model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error loading model: ${error.message}`, 'error');
    }
}

// Unload model
async function unloadModel() {
    try {
        const response = await fetch(`${API_BASE}/unload`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentModel = null;
            showToast('Model unloaded', 'success');
            updateModelStatus();
        }
    } catch (error) {
        showToast(`Error unloading model: ${error.message}`, 'error');
    }
}

// Update model status
async function updateModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        const statusText = document.getElementById('status-text');
        const statusDot = document.querySelector('.status-dot');
        const unloadBtn = document.getElementById('unload-btn');

        if (data.success && data.status.loaded) {
            statusText.textContent = `Loaded: ${data.status.model_id} (${data.status.device})`;
            statusDot.classList.add('active');
            unloadBtn.style.display = 'block';
            unloadBtn.onclick = unloadModel;
            currentModel = data.status.model_id;
        } else {
            statusText.textContent = 'No model loaded';
            statusDot.classList.remove('active');
            unloadBtn.style.display = 'none';
            currentModel = null;
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Chat functionality
function initializeChat() {
    const sendBtn = document.getElementById('send-btn');
    const chatInput = document.getElementById('chat-input');
    const maxLengthSlider = document.getElementById('max-length');
    const temperatureSlider = document.getElementById('temperature');

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Update slider values
    maxLengthSlider.addEventListener('input', (e) => {
        document.getElementById('max-length-value').textContent = e.target.value;
    });

    temperatureSlider.addEventListener('input', (e) => {
        document.getElementById('temperature-value').textContent = e.target.value;
    });
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    const prompt = chatInput.value.trim();

    if (!prompt) return;

    if (!currentModel) {
        showToast('Please load a model first', 'error');
        return;
    }

    // Add user message
    addMessage('user', prompt);
    chatInput.value = '';

    // Disable input
    sendBtn.disabled = true;
    chatInput.disabled = true;

    try {
        const maxLength = parseInt(document.getElementById('max-length').value);
        const temperature = parseFloat(document.getElementById('temperature').value);

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                max_length: maxLength,
                temperature: temperature
            })
        });

        const data = await response.json();

        if (data.success) {
            addMessage('assistant', data.response);
        } else {
            addMessage('error', `Error: ${data.error}`);
        }
    } catch (error) {
        addMessage('error', `Error: ${error.message}`);
    }

    // Re-enable input
    sendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
}

function addMessage(type, content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num;
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}
