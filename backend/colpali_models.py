import json
import base64
from typing import List, Optional, Dict, Union
import requests
from PIL import Image
import io

class ColPaliClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def get_embeddings(self, images_encoded: Optional[List[str]] = None, queries: Optional[List[str]] = None) -> Dict:
        if not images_encoded and not queries:
            raise ValueError("At least one of images or queries must be provided for embedding")
        data = {}
        
        if images_encoded:
            data["images"] = images_encoded
            
        if queries:
            data["queries"] = queries

        try:
            response = requests.post(
                self.endpoint_url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to ColPali endpoint: {str(e)}")