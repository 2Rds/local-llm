import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, list_models, model_info
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "models_config.json"
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_id = None

        # Initialize config file
        if not self.config_file.exists():
            self._save_config({})

    def _load_config(self):
        """Load models configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_config(self, config):
        """Save models configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def search_models(self, query="", limit=20, task="text-generation"):
        """Search models on HuggingFace"""
        try:
            models = list_models(
                search=query,
                task=task,
                sort="downloads",
                direction=-1,
                limit=limit
            )

            results = []
            for model in models:
                try:
                    info = model_info(model.modelId)
                    results.append({
                        "id": model.modelId,
                        "name": model.modelId.split('/')[-1],
                        "author": model.modelId.split('/')[0] if '/' in model.modelId else "unknown",
                        "downloads": getattr(model, 'downloads', 0),
                        "likes": getattr(model, 'likes', 0),
                        "tags": getattr(model, 'tags', []),
                    })
                except:
                    continue

            return results
        except Exception as e:
            print(f"Error searching models: {e}")
            return []

    def download_model(self, model_id, progress_callback=None):
        """Download a model from HuggingFace"""
        try:
            model_path = self.models_dir / model_id.replace('/', '--')

            # Download model
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )

            # Update config
            config = self._load_config()
            config[model_id] = {
                "path": str(model_path),
                "downloaded_at": str(Path(model_path).stat().st_mtime)
            }
            self._save_config(config)

            return {"success": True, "path": str(model_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_downloaded_models(self):
        """List all downloaded models"""
        config = self._load_config()
        models = []

        for model_id, model_data in config.items():
            model_path = Path(model_data["path"])
            if model_path.exists():
                # Get size
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                models.append({
                    "id": model_id,
                    "name": model_id.split('/')[-1],
                    "author": model_id.split('/')[0] if '/' in model_id else "unknown",
                    "path": str(model_path),
                    "size": total_size,
                    "size_mb": round(total_size / (1024 * 1024), 2)
                })

        return models

    def delete_model(self, model_id):
        """Delete a downloaded model"""
        try:
            config = self._load_config()
            if model_id in config:
                model_path = Path(config[model_id]["path"])
                if model_path.exists():
                    shutil.rmtree(model_path)
                del config[model_id]
                self._save_config(config)

                # Unload if currently loaded
                if self.current_model_id == model_id:
                    self.unload_model()

                return {"success": True}
            return {"success": False, "error": "Model not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_model(self, model_id):
        """Load a model for inference"""
        try:
            # Unload current model if any
            if self.loaded_model is not None:
                self.unload_model()

            config = self._load_config()
            if model_id not in config:
                return {"success": False, "error": "Model not downloaded"}

            model_path = config[model_id]["path"]

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Load model
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            if device == "cpu":
                self.loaded_model = self.loaded_model.to(device)

            self.current_model_id = model_id

            return {
                "success": True,
                "model_id": model_id,
                "device": device
            }
        except Exception as e:
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.current_model_id = None
            return {"success": False, "error": str(e)}

    def unload_model(self):
        """Unload the current model"""
        if self.loaded_model is not None:
            del self.loaded_model
            del self.loaded_tokenizer
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.current_model_id = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """Generate text using the loaded model"""
        if self.loaded_model is None or self.loaded_tokenizer is None:
            return {"success": False, "error": "No model loaded"}

        try:
            # Tokenize input
            inputs = self.loaded_tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.loaded_model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.loaded_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.loaded_tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the output
            response = generated_text[len(prompt):].strip()

            return {
                "success": True,
                "response": response,
                "full_text": generated_text
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_status(self):
        """Get current model status"""
        return {
            "loaded": self.loaded_model is not None,
            "model_id": self.current_model_id,
            "device": str(self.loaded_model.device) if self.loaded_model is not None else None
        }
