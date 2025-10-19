# app.py - Qwen3-VL Modal Deployment (Compatible with Bot)
"""
UPDATED: Compatible dengan Smart AI Telegram Bot
Changes:
1. Endpoint renamed to match bot expectations
2. Response format matches bot's OCR expectations
3. Added backward compatibility mode
"""

import modal
from modal import Image
from pydantic import BaseModel
import io
import base64
import torch
from PIL import Image as PILImage

# --- Modal App ---
app = modal.App("qwen2vl-api")

# --- Image Environment ---
qwen_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision",
        "transformers>=4.45",
        "accelerate>=0.30",
        "pillow>=10.0",
        "fastapi",
        "pydantic",
        "qwen-vl-utils>=0.0.14"
    )
)

# --- Model Class ---
@app.cls(
    gpu="T4",
    image=qwen_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    max_containers=2,  # Increased for better concurrent handling
)
class Qwen3VLModel:
    @modal.enter()
    def load_model(self):
        """Initialize model when container starts"""
        import os
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        model_name = "Qwen/Qwen3-VL-4B-Thinking"
        hf_token = os.environ.get("HF_TOKEN")
        
        try:
            print("🔄 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            print("🔄 Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                token=hf_token,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            print("✅ Qwen3-VL-4B-Thinking loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    @modal.method()
    def generate(self, image_bytes: bytes, prompt: str, max_tokens: int = 2048) -> str:
        """Generate response from image and prompt"""
        try:
            from qwen_vl_utils import process_vision_info
            
            # Load and validate image
            image = PILImage.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                do_resize=False
            )
            
            # Move to GPU
            inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            
            # Decode
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Cleanup
            del inputs, output_ids, generated_ids
            torch.cuda.empty_cache()
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ Error in generate: {str(e)}")
            raise


# --- Request Model ---
class VisionRequest(BaseModel):
    """Unified request model compatible with both formats"""
    image_b64: str
    prompt: str
    max_tokens: int = 2048


# --- BACKWARD COMPATIBLE ENDPOINT (Old bot format) ---
@app.function(image=qwen_image)
@modal.web_endpoint(method="POST")
async def analyze(request: dict):
    """
    BACKWARD COMPATIBLE endpoint for existing bot.
    
    Expected format:
    {
        "image_b64": "base64_string",
        "prompt": "analyze this image",
        "max_tokens": 2048
    }
    
    Returns:
    {
        "success": true,
        "response": "analysis text",
        "content": "analysis text"  # for compatibility
    }
    """
    try:
        # Extract parameters
        image_b64 = request.get("image_b64", "")
        prompt = request.get("prompt", "Analyze this image in detail.")
        max_tokens = request.get("max_tokens", 2048)
        
        # Validate
        if not image_b64:
            return {
                "success": False,
                "error": "image_b64 is required"
            }
        
        # Decode base64
        try:
            image_data = base64.b64decode(image_b64)
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid base64: {str(e)}"
            }
        
        # Validate image
        try:
            PILImage.open(io.BytesIO(image_data))
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }
        
        # Generate response
        model = Qwen3VLModel()
        result = model.generate.remote(
            image_data,
            prompt,
            max_tokens
        )
        
        # Return in BOTH formats for compatibility
        return {
            "success": True,
            "response": result,     # New format
            "content": result       # Old format (for backward compatibility)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# --- NEW VQA ENDPOINT (Standard format) ---
@app.function(image=qwen_image)
@modal.web_endpoint(method="POST")
async def vqa(request: VisionRequest):
    """
    Standard VQA endpoint with Pydantic validation.
    
    Example:
    ```bash
    curl -X POST https://your-app--qwen3vl-api-vqa.modal.run \
      -H "Content-Type: application/json" \
      -d '{
        "image_b64": "base64_encoded_image",
        "prompt": "What is in this image?",
        "max_tokens": 2048
      }'
    ```
    """
    try:
        # Decode and validate
        image_data = base64.b64decode(request.image_b64)
        PILImage.open(io.BytesIO(image_data))
        
        # Generate
        model = Qwen3VLModel()
        result = model.generate.remote(
            image_data,
            request.prompt,
            request.max_tokens
        )
        
        return {
            "success": True,
            "response": result,
            "prompt": request.prompt,
            "model": "Qwen3-VL-4B-Thinking"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# --- HEALTH CHECK ---
@app.function(image=qwen_image)
@modal.web_endpoint(method="GET")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qwen3vl-api",
        "model": "Qwen3-VL-4B-Thinking",
        "endpoints": {
            "analyze": "/analyze (POST) - Backward compatible",
            "vqa": "/vqa (POST) - Standard format",
            "health": "/health (GET) - This endpoint"
        }
    }


# --- DEPLOYMENT INFO ---
@app.local_entrypoint()
def info():
    """Print deployment information"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Qwen3-VL-4B-Thinking API - Modal Deployment             ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🤖 Model: Qwen/Qwen3-VL-4B-Thinking
    🎯 GPU: NVIDIA T4
    ⚡ Max Containers: 2
    
    📡 Endpoints:
    ├─ POST /analyze    - Backward compatible (old bot format)
    ├─ POST /vqa        - Standard VQA format
    └─ GET  /health     - Health check
    
    🚀 Deploy:
       modal deploy app.py
    
    🔍 Test:
       curl https://your-username--qwen3vl-api-health.modal.run
    
    📊 Monitor:
       modal app logs qwen3vl-api
    """)
