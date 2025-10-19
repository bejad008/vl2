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
        "transformers>=4.40",
        "accelerate>=0.30",
        "pillow",
        "fastapi"
    )
)

# --- Model Class ---
@app.cls(
    gpu="T4",
    image=qwen_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
class Qwen2VLModel:
    @modal.enter()
    def load_model(self):
        """Initialize model when container starts"""
        import os
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        model_name = "Qwen/Qwen3-VL-4B-Thinking"
        hf_token = os.environ.get("HF_TOKEN")
        
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

    @modal.method()
    def generate(self, image_bytes: bytes, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from image and prompt"""
        try:
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
            text_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Move to GPU
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode only generated tokens
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
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
        model = Qwen2VLModel()
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
