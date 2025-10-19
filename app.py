# app.py
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
        "transformers==4.57.0",
        "accelerate>=0.30",
        "pillow>=10.0",
        "fastapi",
        "pydantic",
        "qwen-vl-utils>=0.0.14"
    )
    .apt_install("libssl-dev", "libffi-dev")
)

# --- Model Class ---
@app.cls(
    gpu="T4",
    image=qwen_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    allow_concurrent_inputs=1,
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
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            print("Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    @modal.method()
    def generate(self, image_bytes: bytes, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from image and prompt"""
        try:
            from qwen_vl_utils import process_vision_info
            
            # Load and validate image
            image = PILImage.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Prepare messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info using qwen-vl-utils
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            # Process inputs with do_resize=False (qwen-vl-utils already resized)
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
            
            # Decode only generated tokens
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
            print(f"Error in generate: {str(e)}")
            raise


# --- Request Model ---
class VQARequest(BaseModel):
    image_b64: str
    prompt: str
    max_tokens: int = 512
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_b64": "iVBORw0KGgoAAAANS...",
                "prompt": "What is in this image?",
                "max_tokens": 512
            }
        }


# --- Web Endpoint (FastAPI) ---
@app.function(
    image=qwen_image,
)
@modal.web_endpoint(method="POST")
async def vqa(request: VQARequest):
    """
    Visual Question Answering endpoint
    
    Example curl:
    ```bash
    curl -X POST https://your-username--qwen2vl-api-vqa.modal.run \
      -H "Content-Type: application/json" \
      -d '{
        "image_b64": "base64_encoded_image_here",
        "prompt": "Describe this image in detail",
        "max_tokens": 512
      }'
    ```
    """
    try:
        # Validate base64
        try:
            image_data = base64.b64decode(request.image_b64)
        except Exception as e:
            return {
                "error": "Invalid base64 image data",
                "details": str(e)
            }
        
        # Validate image can be opened
        try:
            PILImage.open(io.BytesIO(image_data))
        except Exception as e:
            return {
                "error": "Invalid image format",
                "details": str(e)
            }
        
        # Generate response using the model class
        model = Qwen3VLModel()
        result = model.generate.remote(
            image_data, 
            request.prompt,
            request.max_tokens
        )
        
        return {
            "success": True,
            "response": result,
            "prompt": request.prompt
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# --- Health Check Endpoint ---
@app.function(image=qwen_image)
@modal.web_endpoint(method="GET")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "qwen2vl-api"}
