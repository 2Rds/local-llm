import os
import json
import shutil
import base64
from io import BytesIO
from pathlib import Path
from huggingface_hub import snapshot_download, list_models, model_info
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import cv2
import numpy as np

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "models_config.json"
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.loaded_pipeline = None
        self.current_model_id = None
        self.current_model_type = None

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

    def _detect_model_type(self, model_id):
        """Detect the type of model from HuggingFace"""
        try:
            info = model_info(model_id)
            pipeline_tag = getattr(info, 'pipeline_tag', None)

            # Map HuggingFace pipeline tags to our model types
            type_mapping = {
                'text-generation': 'text-to-text',
                'text2text-generation': 'text-to-text',
                'text-to-image': 'text-to-image',
                'image-to-image': 'image-to-image',
                'text-to-video': 'text-to-video',
                'image-to-video': 'image-to-video',
            }

            model_type = type_mapping.get(pipeline_tag, 'text-to-text')
            return model_type
        except Exception as e:
            print(f"Could not detect model type, defaulting to text-to-text: {e}")
            return 'text-to-text'

    def search_models(self, query="", limit=20, model_type=None):
        """Search models on HuggingFace"""
        try:
            # Map our model types to HuggingFace pipeline tags
            task_mapping = {
                'text-to-text': 'text-generation',
                'text-to-image': 'text-to-image',
                'image-to-image': 'image-to-image',
                'text-to-video': 'text-to-video',
                'image-to-video': 'image-to-video',
            }

            task = task_mapping.get(model_type) if model_type else None

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
                    detected_type = self._detect_model_type(model.modelId)

                    results.append({
                        "id": model.modelId,
                        "name": model.modelId.split('/')[-1],
                        "author": model.modelId.split('/')[0] if '/' in model.modelId else "unknown",
                        "downloads": getattr(model, 'downloads', 0),
                        "likes": getattr(model, 'likes', 0),
                        "tags": getattr(model, 'tags', []),
                        "model_type": detected_type,
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
            model_type = self._detect_model_type(model_id)

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
                "downloaded_at": str(Path(model_path).stat().st_mtime),
                "model_type": model_type
            }
            self._save_config(config)

            return {"success": True, "path": str(model_path), "model_type": model_type}
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
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "model_type": model_data.get("model_type", "text-to-text")
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
            if self.loaded_model is not None or self.loaded_pipeline is not None:
                self.unload_model()

            config = self._load_config()
            if model_id not in config:
                return {"success": False, "error": "Model not downloaded"}

            model_path = config[model_id]["path"]
            model_type = config[model_id].get("model_type", "text-to-text")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load based on model type
            if model_type == "text-to-text":
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

            elif model_type == "text-to-image":
                # Load Stable Diffusion or similar
                try:
                    self.loaded_pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        safety_checker=None,
                        trust_remote_code=True
                    )
                    self.loaded_pipeline = self.loaded_pipeline.to(device)
                except Exception as e:
                    # Fallback: Try with StableDiffusionPipeline explicitly
                    try:
                        self.loaded_pipeline = StableDiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                            safety_checker=None,
                            trust_remote_code=True
                        )
                        self.loaded_pipeline = self.loaded_pipeline.to(device)
                    except Exception as e2:
                        raise Exception(f"Unable to load text-to-image model. Error: {str(e2)}")

            elif model_type == "image-to-image":
                # Load image-to-image pipeline
                try:
                    self.loaded_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        safety_checker=None,
                        trust_remote_code=True
                    )
                    self.loaded_pipeline = self.loaded_pipeline.to(device)
                except Exception as e:
                    # Fallback: Try generic DiffusionPipeline
                    try:
                        self.loaded_pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                            safety_checker=None,
                            trust_remote_code=True
                        )
                        self.loaded_pipeline = self.loaded_pipeline.to(device)
                    except Exception as e2:
                        raise Exception(f"Unable to load image-to-image model. Error: {str(e2)}")

            elif model_type in ["text-to-video", "image-to-video"]:
                # Load video generation pipeline
                # Try multiple loading strategies for different model formats
                loaded = False
                last_error = None

                # Strategy 1: Try standard DiffusionPipeline
                try:
                    from pathlib import Path
                    model_index_path = Path(model_path) / "model_index.json"

                    if model_index_path.exists():
                        self.loaded_pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                            trust_remote_code=True
                        )
                        self.loaded_pipeline = self.loaded_pipeline.to(device)
                        loaded = True
                except Exception as e:
                    last_error = e

                # Strategy 2: Try loading with custom pipeline
                if not loaded:
                    try:
                        # Some video models use custom pipeline classes
                        self.loaded_pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                            trust_remote_code=True,
                            custom_pipeline=model_path
                        )
                        self.loaded_pipeline = self.loaded_pipeline.to(device)
                        loaded = True
                    except Exception as e:
                        last_error = e

                # Strategy 3: Try loading as a generic transformers pipeline
                if not loaded:
                    try:
                        self.loaded_pipeline = pipeline(
                            task="image-to-video" if model_type == "image-to-video" else "text-to-video",
                            model=model_path,
                            trust_remote_code=True,
                            device=0 if device == "cuda" else -1
                        )
                        loaded = True
                    except Exception as e:
                        last_error = e

                if not loaded:
                    # Provide helpful error message
                    error_msg = f"Unable to load video model. This model may not be compatible with standard video pipelines. "
                    error_msg += f"The model might require a specific loading method or custom code. "
                    error_msg += f"Last error: {str(last_error)}"
                    raise Exception(error_msg)

            self.current_model_id = model_id
            self.current_model_type = model_type

            return {
                "success": True,
                "model_id": model_id,
                "model_type": model_type,
                "device": device
            }
        except Exception as e:
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.loaded_pipeline = None
            self.current_model_id = None
            self.current_model_type = None
            return {"success": False, "error": str(e)}

    def unload_model(self):
        """Unload the current model"""
        if self.loaded_model is not None:
            del self.loaded_model
            del self.loaded_tokenizer
        if self.loaded_pipeline is not None:
            del self.loaded_pipeline

        self.loaded_model = None
        self.loaded_tokenizer = None
        self.loaded_pipeline = None
        self.current_model_id = None
        self.current_model_type = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50,
                 num_inference_steps=50, guidance_scale=7.5, image_data=None):
        """Generate output using the loaded model"""

        if self.current_model_type is None:
            return {"success": False, "error": "No model loaded"}

        try:
            if self.current_model_type == "text-to-text":
                return self._generate_text(prompt, max_length, temperature, top_p, top_k)

            elif self.current_model_type == "text-to-image":
                return self._generate_image_from_text(prompt, num_inference_steps, guidance_scale)

            elif self.current_model_type == "image-to-image":
                if not image_data:
                    return {"success": False, "error": "Image input required for image-to-image model"}
                return self._generate_image_from_image(prompt, image_data, num_inference_steps, guidance_scale)

            elif self.current_model_type == "text-to-video":
                return self._generate_video_from_text(prompt, num_inference_steps, guidance_scale)

            elif self.current_model_type == "image-to-video":
                if not image_data:
                    return {"success": False, "error": "Image input required for image-to-video model"}
                return self._generate_video_from_image(prompt, image_data, num_inference_steps)

            else:
                return {"success": False, "error": f"Unsupported model type: {self.current_model_type}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_text(self, prompt, max_length, temperature, top_p, top_k):
        """Generate text using text-generation model"""
        if self.loaded_model is None or self.loaded_tokenizer is None:
            return {"success": False, "error": "No text model loaded"}

        inputs = self.loaded_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.loaded_model.device) for k, v in inputs.items()}

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

        generated_text = self.loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()

        return {
            "success": True,
            "response": response,
            "full_text": generated_text,
            "type": "text"
        }

    def _generate_image_from_text(self, prompt, num_inference_steps, guidance_scale):
        """Generate image from text using text-to-image model"""
        if self.loaded_pipeline is None:
            return {"success": False, "error": "No image generation model loaded"}

        result = self.loaded_pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        image = result.images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "image": img_str,
            "type": "image"
        }

    def _generate_image_from_image(self, prompt, image_data, num_inference_steps, guidance_scale):
        """Generate image from image using image-to-image model"""
        if self.loaded_pipeline is None:
            return {"success": False, "error": "No image-to-image model loaded"}

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        result = self.loaded_pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=0.75
        )

        image = result.images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "image": img_str,
            "type": "image"
        }

    def _generate_video_from_text(self, prompt, num_inference_steps, guidance_scale):
        """Generate video from text"""
        if self.loaded_pipeline is None:
            return {"success": False, "error": "No video generation model loaded"}

        try:
            result = self.loaded_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=16  # Generate 16 frames
            )

            # result.frames is a list of PIL images
            frames = result.frames[0] if hasattr(result, 'frames') else result.images

            # Convert frames to video (save as MP4)
            video_path = self.models_dir / f"temp_video_{hash(prompt)}.mp4"

            # Convert PIL images to numpy arrays
            frame_arrays = [np.array(frame) for frame in frames]

            # Save as video
            height, width, _ = frame_arrays[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 8.0, (width, height))

            for frame in frame_arrays:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()

            # Read video and convert to base64
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_str = base64.b64encode(video_bytes).decode()

            # Clean up
            video_path.unlink()

            return {
                "success": True,
                "video": video_str,
                "type": "video"
            }
        except Exception as e:
            return {"success": False, "error": f"Video generation error: {str(e)}"}

    def _generate_video_from_image(self, prompt, image_data, num_inference_steps):
        """Generate video from image"""
        if self.loaded_pipeline is None:
            return {"success": False, "error": "No image-to-video model loaded"}

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            result = self.loaded_pipeline(
                prompt=prompt,
                image=input_image,
                num_inference_steps=num_inference_steps,
                num_frames=16
            )

            frames = result.frames[0] if hasattr(result, 'frames') else result.images

            # Convert frames to video
            video_path = self.models_dir / f"temp_video_{hash(prompt)}.mp4"

            frame_arrays = [np.array(frame) for frame in frames]
            height, width, _ = frame_arrays[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 8.0, (width, height))

            for frame in frame_arrays:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()

            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_str = base64.b64encode(video_bytes).decode()

            video_path.unlink()

            return {
                "success": True,
                "video": video_str,
                "type": "video"
            }
        except Exception as e:
            return {"success": False, "error": f"Video generation error: {str(e)}"}

    def get_model_status(self):
        """Get current model status"""
        return {
            "loaded": self.loaded_model is not None or self.loaded_pipeline is not None,
            "model_id": self.current_model_id,
            "model_type": self.current_model_type,
            "device": str(self.loaded_model.device) if self.loaded_model is not None else (
                str(self.loaded_pipeline.device) if self.loaded_pipeline is not None else None
            )
        }
