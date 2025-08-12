from qdrant_client import QdrantClient, models
import stamina
from tqdm import tqdm
from helper_functions import image_to_base64, load_images, encode_images_base64
from colpali_models import ColPaliClient
import uuid
from pathlib import Path
import re
from pathlib import Path

def create_qdrant_client(qdrant_uri):
    qdrant_client = QdrantClient(
        url=qdrant_uri
    )
    return qdrant_client

def delete_qdrant_collection(qdrant_uri, collection_name):
   qdrant_client = create_qdrant_client(qdrant_uri)
   return qdrant_client.delete_collection(collection_name=collection_name)


def list_qdrant_collections(qdrant_uri):
    qdrant_client = create_qdrant_client(qdrant_uri)
    
    return qdrant_client.get_collections()


def create_qdrant_collection(qdrant_uri, collection_name, vector_size, indexing_threshold):
    try:
        qdrant_client = create_qdrant_client(qdrant_uri)

        qdrant_client.create_collection(
            collection_name=collection_name,
            on_disk_payload=True,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold
            ),
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        )

        return True
    except Exception as e:
        print("Error creating collection in qdrant. {}".format(str(e)))
        raise False
    

@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(points, collection_name, qdrant_client):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
        return True
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False


def index_images_to_qdrant(images_paths, batch_size, collection_name, qdrant_uri, colpali_url, starting_id=0):
    global_id = starting_id

    qdrant_client = create_qdrant_client(qdrant_uri)
    colpali_client = ColPaliClient(colpali_url)

    with tqdm(total=len(images_paths), desc="Indexing Progress") as pbar:
        for i in range(0, len(images_paths), batch_size):
            print("Processing batch")

            batch = images_paths[i : i + batch_size]
            # batch_images = load_images(batch)
            batch_images_encoded = encode_images_base64(batch)

            # Retrieve Embeddings for the image using colpali model
            embedding_results = colpali_client.get_embeddings(images_encoded=batch_images_encoded)
            print("working for images")
            image_embeddings = embedding_results["image_embeddings"]

            # prepare points for Qdrant
            points = []
            print("Creating points")
            for j, embedding in enumerate(image_embeddings):
                # Convert the file path to a Path object
                file_path = Path(batch[j])
                # Get the file stem (filename without extension)
                file_stem = file_path.stem
                # Use regex to extract the page number (e.g., matching '_page-0015')
                match = re.search(r'_page-(\d+)', file_stem)
                if match:
                    page_number = int(match.group(1))
                else:
                    page_number = None  # Or assign a default value if no page number is found

                points.append(
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=embedding,
                        payload={
                            "image": file_path.as_posix(),
                            "page_number": page_number,
                            "ISBN" : ""
                        },
                    )
                )
                global_id += 1

            try:
                upsert_to_qdrant(points, collection_name, qdrant_client)
                print("Upsert done for batch")
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue
            pbar.update(batch_size)
    print("Indexing complete!")


def search_qdrant(collection_name, user_query, qdrant_uri, colpali_url, top_k):
    try:
        qdrant_client = create_qdrant_client(qdrant_uri)
        colpali_client = ColPaliClient(colpali_url)

        # Retrieve Embeddings for the image using colpali model
        embedding_results = colpali_client.get_embeddings(queries=[user_query])
        query_embeddings = embedding_results["query_embeddings"]

        # Perform the query on Qdrant, searching for the most similar points (multivector query)
        search_result = qdrant_client.query_points(
            collection_name=collection_name, 
            query=query_embeddings[0], 
            with_payload=["image","ISBN","page_number"],
            limit=top_k
        )
        
        return search_result
    
    except Exception as e:
        raise RuntimeError(f"Error during Qdrant search: {str(e)}")