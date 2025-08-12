import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from helper_functions import convert_pdf_to_images, extract_images_and_names_from_zip, create_hash_folder
from qdrant_models import index_images_to_qdrant, search_qdrant, create_qdrant_collection, list_qdrant_collections, delete_qdrant_collection

load_dotenv()
QDRANT_URI = os.getenv("QDRANT_URI")
COLPALI_URI = os.getenv("COLPALI_URI")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# directory to save input files
BASE_UPLOAD_DIRECTORY = os.getenv('BASE_UPLOAD_DIR', '') # update this base directory

# creating base directory if it does not exists
Path(BASE_UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

class ImageRetrievalRequest(BaseModel):
    user_query: str

class QdrantCollectionCreate(BaseModel):
    collection_name: str
    vector_size: int
    indexing_threshold: int

# class QdrantCollectionDelete(BaseModel):
#     collection_name: str

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to create collection in qdrant
@app.post("/create_qdrant_collection")
async def qdrant_create_collection(request: QdrantCollectionCreate):
    return create_qdrant_collection(QDRANT_URI, request.collection_name, request.vector_size, request.indexing_threshold)

# Endpoint to delete collection in qdrant
# @app.post("/delete_qdrant_collection")
# async def qdrant_delete_collection(request: QdrantCollectionDelete):
#     return delete_qdrant_collection(QDRANT_URI, request.collection_name)

# Endpoint to delete collection in qdrant
@app.post("/get_qdrant_collections")
async def qdrant_list_collections():
    return list_qdrant_collections(QDRANT_URI)


# ENDPOINT TO INDEX THE IMAGES TO QDRANT
@app.post("/document_embed")
async def embed_index_documents(file: UploadFile = File(...)):
    try:
        # Create a unique hash-based folder
        hash_folder, images_folder = create_hash_folder(BASE_UPLOAD_DIRECTORY)
        hash_folder = Path(hash_folder)
        images_folder = Path(images_folder)
        filename = file.filename

        # Save the uploaded file in the hash folder
        file_location = hash_folder / filename
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        if filename.endswith(".pdf"):
            images = convert_pdf_to_images(str(file_location))
            pdf_filename = filename.rsplit(".", 1)[0]
            for idx, image in enumerate(images):
                image_path = images_folder / f"{pdf_filename}_page_{idx + 1}.png"
                image.save(str(image_path))
        
        elif filename.endswith(".zip"):
            images, image_names = extract_images_and_names_from_zip(str(file_location))
            for image, name in zip(images, image_names):
                image_path = images_folder / name
                image.save(str(image_path))
        
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = images_folder / filename
            with open(image_path, "wb") as f:
                f.write(content)
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        image_files = [str(images_folder / img) for img in os.listdir(str(images_folder))]
        print("Input file processed. Calling Embedding function Now!")
        print("Number of images:{}".format(len(image_files)))
        index_images_to_qdrant(image_files, batch_size=4, collection_name=QDRANT_COLLECTION_NAME, qdrant_uri=QDRANT_URI, colpali_url=COLPALI_URI)
        return {"status": "Document embedded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding documents: {str(e)}")
    
    # finally:
    #     if os.path.exists(hash_folder):
    #         shutil.rmtree(hash_folder)


# ENDPOINT TO RETRIEVE TOP K RELEVANT TEXTBOOK PAGE IMAGES
@app.post("/document_retrieval")
async def get_relevant_documents(request: ImageRetrievalRequest):
    try:
        search_result = search_qdrant(QDRANT_COLLECTION_NAME, request.user_query, qdrant_uri=QDRANT_URI, colpali_url=COLPALI_URI, top_k=15)
    
        if not search_result.points:
            raise HTTPException(status_code=404, detail="No matching images found")
        
        scores = [r.score for r in search_result.points]
        
        image_points = [r.payload for r in search_result.points]

        retrieved_points_with_scores = []
        for point, score in zip(image_points, scores):
            point_with_score = point.copy()
            point_with_score['score'] = score
            retrieved_points_with_scores.append(point_with_score)

        response = {
            "retrieved_image_points": retrieved_points_with_scores
        }
        print("Relevant images retreived")

        return response

        # return FileResponse(row_image_paths[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text embedding: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)