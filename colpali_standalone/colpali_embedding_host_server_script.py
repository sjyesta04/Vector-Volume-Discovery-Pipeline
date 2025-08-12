import io
import os
import base64
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import HfFolder
from typing import List, Optional
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import logging
from contextlib import asynccontextmanager

model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    
    try:
        model_id = "vidore/colpali-v1.2"
        device = "cuda:0"
        dtype = torch.float16
        
        model = ColPali.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_id)
        yield
        
    except Exception as e:
        raise RuntimeError(f"Error loading model or processor: {str(e)}")
    
    finally:
        if model:
            del model
        if processor:
            del processor

app = FastAPI(lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    images: Optional[List[str]] = None
    queries: Optional[List[str]] = None

class EmbeddingResponse(BaseModel):
    image_embeddings: Optional[List[List[List[float]]]] = None
    query_embeddings: Optional[List[List[List[float]]]] = None

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        image_embeddings = None
        query_embeddings = None

        # if request contains images
        if request.images:
            print("Images exists!")
            images = []
            for img_str in request.images:
                img_bytes = base64.b64decode(img_str)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)

            print("Images decoded!")
            
            batch_images = processor.process_images(images).to(model.device)
            with torch.no_grad():
                image_embeddings = model(**batch_images)
                image_embeddings = image_embeddings.cpu().float().numpy().tolist()

            print("Created embeddings for images!")

        # if request contains queries
        if request.queries:
            print("Queries exists!")
            with torch.no_grad():
                batch_query = processor.process_queries(request.queries).to(
                    model.device
                )
                query_embeddings = model(**batch_query)
                query_embeddings = query_embeddings.cpu().float().numpy().tolist()

                print("Query embeddings created!!")

        if image_embeddings:
            print(f"Image Embedding dimensions: {len(image_embeddings)}, {len(image_embeddings[0])}, {len(image_embeddings[0][0])}")

        if query_embeddings:
            print(f"Query Embedding dimensions: {len(query_embeddings)}, {len(query_embeddings[0])}, {len(query_embeddings[0][0])}")
        
        return EmbeddingResponse(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)