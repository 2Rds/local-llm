from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from model_manager import ModelManager
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Initialize model manager
model_manager = ModelManager(models_dir="models")

# Serve frontend
frontend_dir = Path(__file__).parent.parent / "frontend"

@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(frontend_dir, path)

# API Routes

@app.route('/api/search', methods=['GET'])
def search_models():
    """Search for models on HuggingFace"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))

    results = model_manager.search_models(query=query, limit=limit)
    return jsonify({"success": True, "models": results})

@app.route('/api/download', methods=['POST'])
def download_model():
    """Download a model from HuggingFace"""
    data = request.json
    model_id = data.get('model_id')

    if not model_id:
        return jsonify({"success": False, "error": "model_id required"}), 400

    result = model_manager.download_model(model_id)
    return jsonify(result)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List downloaded models"""
    models = model_manager.list_downloaded_models()
    return jsonify({"success": True, "models": models})

@app.route('/api/models/<path:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a downloaded model"""
    # Replace -- back to /
    model_id = model_id.replace('--', '/')
    result = model_manager.delete_model(model_id)
    return jsonify(result)

@app.route('/api/load', methods=['POST'])
def load_model():
    """Load a model for inference"""
    data = request.json
    model_id = data.get('model_id')

    if not model_id:
        return jsonify({"success": False, "error": "model_id required"}), 400

    result = model_manager.load_model(model_id)
    return jsonify(result)

@app.route('/api/unload', methods=['POST'])
def unload_model():
    """Unload the current model"""
    model_manager.unload_model()
    return jsonify({"success": True})

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text using the loaded model"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)

    if not prompt:
        return jsonify({"success": False, "error": "prompt required"}), 400

    result = model_manager.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current model status"""
    status = model_manager.get_model_status()
    return jsonify({"success": True, "status": status})

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"success": True, "message": "Server is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
