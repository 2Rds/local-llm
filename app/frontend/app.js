// API Base URL
const API_BASE = 'http://localhost:5000/api';

// State
let currentModel = null;
let currentModelType = null;
let uploadedImages = {
    transform: null,
    i2v: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeSearch();
    initializeTextChat();
    initializeImageGeneration();
    initializeImageTransform();
    initializeVideoGeneration();
    initializeImageToVideo();
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

// Show/hide tabs based on model type
function updateVisibleTabs(modelType) {
    // Hide all generation tabs
    document.getElementById('text-chat-tab-btn').style.display = 'none';
    document.getElementById('image-gen-tab-btn').style.display = 'none';
    document.getElementById('image-transform-tab-btn').style.display = 'none';
    document.getElementById('video-gen-tab-btn').style.display = 'none';
    document.getElementById('image-to-video-tab-btn').style.display = 'none';

    // Show appropriate tab based on model type
    if (modelType === 'text-to-text') {
        const tab = document.getElementById('text-chat-tab-btn');
        tab.style.display = 'block';
        tab.click();
    } else if (modelType === 'text-to-image') {
        const tab = document.getElementById('image-gen-tab-btn');
        tab.style.display = 'block';
        tab.click();
    } else if (modelType === 'image-to-image') {
        const tab = document.getElementById('image-transform-tab-btn');
        tab.style.display = 'block';
        tab.click();
    } else if (modelType === 'text-to-video') {
        const tab = document.getElementById('video-gen-tab-btn');
        tab.style.display = 'block';
        tab.click();
    } else if (modelType === 'image-to-video') {
        const tab = document.getElementById('image-to-video-tab-btn');
        tab.style.display = 'block';
        tab.click();
    }
}

// Search functionality
function initializeSearch() {
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');
    const directDownloadBtn = document.getElementById('direct-download-btn');
    const directInput = document.getElementById('direct-input');
    const modelTypeFilter = document.getElementById('model-type-filter');
    const suggestions = document.querySelectorAll('.model-suggestion');

    // Direct download functionality
    directDownloadBtn.addEventListener('click', () => {
        const modelId = directInput.value.trim();
        if (modelId) {
            downloadModel(modelId);
            directInput.value = '';
        } else {
            showToast('Please enter a model ID', 'error');
        }
    });

    directInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const modelId = directInput.value.trim();
            if (modelId) {
                downloadModel(modelId);
                directInput.value = '';
            }
        }
    });

    // Search functionality
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // Model type filter
    modelTypeFilter.addEventListener('change', performSearch);

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
    const modelType = document.getElementById('model-type-filter').value;
    const resultsDiv = document.getElementById('search-results');
    const loadingDiv = document.getElementById('search-loading');

    resultsDiv.innerHTML = '';
    loadingDiv.style.display = 'block';

    try {
        let url = `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=20`;
        if (modelType) {
            url += `&model_type=${encodeURIComponent(modelType)}`;
        }

        const response = await fetch(url);
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

    const modelTypeBadge = model.model_type
        ? `<div class="model-type-badge model-type-${model.model_type}">${model.model_type.replace(/-/g, ' ')}</div>`
        : '';

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
        ${modelTypeBadge}
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
            showToast(`Successfully downloaded ${modelId} (Type: ${data.model_type})`, 'success');
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
            currentModelType = data.model_type;
            showToast(`Successfully loaded ${modelId} (${data.model_type}) on ${data.device}`, 'success');
            updateModelStatus();
            updateVisibleTabs(data.model_type);
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
            currentModelType = null;
            showToast('Model unloaded', 'success');
            updateModelStatus();
            updateVisibleTabs(null);
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
            statusText.textContent = `Loaded: ${data.status.model_id} (${data.status.model_type}) [${data.status.device}]`;
            statusDot.classList.add('active');
            unloadBtn.style.display = 'block';
            unloadBtn.onclick = unloadModel;
            currentModel = data.status.model_id;
            currentModelType = data.status.model_type;
            updateVisibleTabs(data.status.model_type);
        } else {
            statusText.textContent = 'No model loaded';
            statusDot.classList.remove('active');
            unloadBtn.style.display = 'none';
            currentModel = null;
            currentModelType = null;
            updateVisibleTabs(null);
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Text Chat functionality
function initializeTextChat() {
    const sendBtn = document.getElementById('text-send-btn');
    const chatInput = document.getElementById('text-input');
    const maxLengthSlider = document.getElementById('text-max-length');
    const temperatureSlider = document.getElementById('text-temperature');

    sendBtn.addEventListener('click', sendTextMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });

    // Update slider values
    maxLengthSlider.addEventListener('input', (e) => {
        document.getElementById('text-max-length-value').textContent = e.target.value;
    });

    temperatureSlider.addEventListener('input', (e) => {
        document.getElementById('text-temperature-value').textContent = e.target.value;
    });
}

async function sendTextMessage() {
    const chatInput = document.getElementById('text-input');
    const chatMessages = document.getElementById('text-messages');
    const sendBtn = document.getElementById('text-send-btn');
    const prompt = chatInput.value.trim();

    if (!prompt) return;

    if (!currentModel || currentModelType !== 'text-to-text') {
        showToast('Please load a text-to-text model first', 'error');
        return;
    }

    // Add user message
    addTextMessage('user', prompt);
    chatInput.value = '';

    // Disable input
    sendBtn.disabled = true;
    chatInput.disabled = true;

    try {
        const maxLength = parseInt(document.getElementById('text-max-length').value);
        const temperature = parseFloat(document.getElementById('text-temperature').value);

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
            addTextMessage('assistant', data.response);
        } else {
            addTextMessage('error', `Error: ${data.error}`);
        }
    } catch (error) {
        addTextMessage('error', `Error: ${error.message}`);
    }

    // Re-enable input
    sendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
}

function addTextMessage(type, content) {
    const chatMessages = document.getElementById('text-messages');
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

// Image Generation (Text-to-Image)
function initializeImageGeneration() {
    const generateBtn = document.getElementById('image-generate-btn');
    const stepsSlider = document.getElementById('image-steps');
    const guidanceSlider = document.getElementById('image-guidance');

    generateBtn.addEventListener('click', generateImage);

    stepsSlider.addEventListener('input', (e) => {
        document.getElementById('image-steps-value').textContent = e.target.value;
    });

    guidanceSlider.addEventListener('input', (e) => {
        document.getElementById('image-guidance-value').textContent = e.target.value;
    });
}

async function generateImage() {
    const prompt = document.getElementById('image-prompt').value.trim();
    const outputDiv = document.getElementById('image-output');
    const generateBtn = document.getElementById('image-generate-btn');

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    if (!currentModel || currentModelType !== 'text-to-image') {
        showToast('Please load a text-to-image model first', 'error');
        return;
    }

    generateBtn.disabled = true;
    outputDiv.innerHTML = '<div class="generation-loading"><div class="spinner"></div><p>Generating image...</p></div>';

    try {
        const steps = parseInt(document.getElementById('image-steps').value);
        const guidance = parseFloat(document.getElementById('image-guidance').value);

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                num_inference_steps: steps,
                guidance_scale: guidance
            })
        });

        const data = await response.json();

        if (data.success && data.image) {
            outputDiv.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Generated image">`;
            showToast('Image generated successfully!', 'success');
        } else {
            outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
        showToast(`Error: ${error.message}`, 'error');
    }

    generateBtn.disabled = false;
}

// Image Transform (Image-to-Image)
function initializeImageTransform() {
    const uploadArea = document.getElementById('transform-upload-area');
    const fileInput = document.getElementById('transform-image-input');
    const preview = document.getElementById('transform-preview');
    const generateBtn = document.getElementById('transform-generate-btn');
    const stepsSlider = document.getElementById('transform-steps');
    const guidanceSlider = document.getElementById('transform-guidance');

    setupImageUpload(uploadArea, fileInput, preview, 'transform');

    generateBtn.addEventListener('click', transformImage);

    stepsSlider.addEventListener('input', (e) => {
        document.getElementById('transform-steps-value').textContent = e.target.value;
    });

    guidanceSlider.addEventListener('input', (e) => {
        document.getElementById('transform-guidance-value').textContent = e.target.value;
    });
}

async function transformImage() {
    const prompt = document.getElementById('transform-prompt').value.trim();
    const outputDiv = document.getElementById('transform-output');
    const generateBtn = document.getElementById('transform-generate-btn');

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    if (!uploadedImages.transform) {
        showToast('Please upload an image first', 'error');
        return;
    }

    if (!currentModel || currentModelType !== 'image-to-image') {
        showToast('Please load an image-to-image model first', 'error');
        return;
    }

    generateBtn.disabled = true;
    outputDiv.innerHTML = '<div class="generation-loading"><div class="spinner"></div><p>Transforming image...</p></div>';

    try {
        const steps = parseInt(document.getElementById('transform-steps').value);
        const guidance = parseFloat(document.getElementById('transform-guidance').value);

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                image_data: uploadedImages.transform,
                num_inference_steps: steps,
                guidance_scale: guidance
            })
        });

        const data = await response.json();

        if (data.success && data.image) {
            outputDiv.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Transformed image">`;
            showToast('Image transformed successfully!', 'success');
        } else {
            outputDiv.innerHTML = '<div class="empty-output">Transformation failed</div>';
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        outputDiv.innerHTML = '<div class="empty-output">Transformation failed</div>';
        showToast(`Error: ${error.message}`, 'error');
    }

    generateBtn.disabled = false;
}

// Video Generation (Text-to-Video)
function initializeVideoGeneration() {
    const generateBtn = document.getElementById('video-generate-btn');
    const stepsSlider = document.getElementById('video-steps');
    const guidanceSlider = document.getElementById('video-guidance');

    generateBtn.addEventListener('click', generateVideo);

    stepsSlider.addEventListener('input', (e) => {
        document.getElementById('video-steps-value').textContent = e.target.value;
    });

    guidanceSlider.addEventListener('input', (e) => {
        document.getElementById('video-guidance-value').textContent = e.target.value;
    });
}

async function generateVideo() {
    const prompt = document.getElementById('video-prompt').value.trim();
    const outputDiv = document.getElementById('video-output');
    const generateBtn = document.getElementById('video-generate-btn');

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    if (!currentModel || currentModelType !== 'text-to-video') {
        showToast('Please load a text-to-video model first', 'error');
        return;
    }

    generateBtn.disabled = true;
    outputDiv.innerHTML = '<div class="generation-loading"><div class="spinner"></div><p>Generating video... This may take several minutes...</p></div>';

    try {
        const steps = parseInt(document.getElementById('video-steps').value);
        const guidance = parseFloat(document.getElementById('video-guidance').value);

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                num_inference_steps: steps,
                guidance_scale: guidance
            })
        });

        const data = await response.json();

        if (data.success && data.video) {
            outputDiv.innerHTML = `<video controls autoplay loop><source src="data:video/mp4;base64,${data.video}" type="video/mp4"></video>`;
            showToast('Video generated successfully!', 'success');
        } else {
            outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
        showToast(`Error: ${error.message}`, 'error');
    }

    generateBtn.disabled = false;
}

// Image to Video
function initializeImageToVideo() {
    const uploadArea = document.getElementById('i2v-upload-area');
    const fileInput = document.getElementById('i2v-image-input');
    const preview = document.getElementById('i2v-preview');
    const generateBtn = document.getElementById('i2v-generate-btn');
    const stepsSlider = document.getElementById('i2v-steps');

    setupImageUpload(uploadArea, fileInput, preview, 'i2v');

    generateBtn.addEventListener('click', generateImageToVideo);

    stepsSlider.addEventListener('input', (e) => {
        document.getElementById('i2v-steps-value').textContent = e.target.value;
    });
}

async function generateImageToVideo() {
    const prompt = document.getElementById('i2v-prompt').value.trim();
    const outputDiv = document.getElementById('i2v-output');
    const generateBtn = document.getElementById('i2v-generate-btn');

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    if (!uploadedImages.i2v) {
        showToast('Please upload an image first', 'error');
        return;
    }

    if (!currentModel || currentModelType !== 'image-to-video') {
        showToast('Please load an image-to-video model first', 'error');
        return;
    }

    generateBtn.disabled = true;
    outputDiv.innerHTML = '<div class="generation-loading"><div class="spinner"></div><p>Generating video... This may take several minutes...</p></div>';

    try {
        const steps = parseInt(document.getElementById('i2v-steps').value);

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                image_data: uploadedImages.i2v,
                num_inference_steps: steps
            })
        });

        const data = await response.json();

        if (data.success && data.video) {
            outputDiv.innerHTML = `<video controls autoplay loop><source src="data:video/mp4;base64,${data.video}" type="video/mp4"></video>`;
            showToast('Video generated successfully!', 'success');
        } else {
            outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        outputDiv.innerHTML = '<div class="empty-output">Generation failed</div>';
        showToast(`Error: ${error.message}`, 'error');
    }

    generateBtn.disabled = false;
}

// Image Upload Helper
function setupImageUpload(uploadArea, fileInput, preview, key) {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageFile(file, preview, key);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file, preview, key);
        } else {
            showToast('Please drop an image file', 'error');
        }
    });
}

function handleImageFile(file, preview, key) {
    const reader = new FileReader();

    reader.onload = (e) => {
        const dataUrl = e.target.result;
        uploadedImages[key] = dataUrl;

        // Show preview
        preview.src = dataUrl;
        preview.style.display = 'block';
        preview.parentElement.querySelector('.upload-placeholder').style.display = 'none';

        showToast('Image uploaded successfully', 'success');
    };

    reader.readAsDataURL(file);
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
