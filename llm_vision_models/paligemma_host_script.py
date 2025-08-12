import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from huggingface_hub import HfFolder

# Hugging Face token login function
def login_to_hf(token):
    HfFolder.save_token(token)

# Request body schema
class ImageQueryRequest(BaseModel):
    image_base64: str
    user_query: str

# Initialize the FastAPI app
app = FastAPI()

# Load the model and processor when the app starts
@app.on_event("startup")
async def load_model():
    global model, processor

    # model_id = "google/paligemma-3b-mix-224"
    model_id = "google/paligemma-3b-pt-448"
    # model_id = "google/paligemma-3b-pt-896"
    device = "cuda:0"
    dtype = torch.float32
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            revision="bfloat16",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Error loading model or processor: {str(e)}")

# Define the prediction endpoint
@app.post("/predict")
async def predict_image(request: ImageQueryRequest):
    try:
        # Decode the base64-encoded image
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
        except (base64.binascii.Error, UnidentifiedImageError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Prepare input for the model
        model_inputs = processor(text=request.user_query, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)

        return {"output": decoded}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Example for Hugging Face token login
HF_TOKEN = "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"
login_to_hf(HF_TOKEN)
