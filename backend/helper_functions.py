import base64
import zipfile
from PIL import Image
from typing import List
from pdf2image import convert_from_path
from io import BytesIO
import uuid
import os
from pathlib import Path

def create_hash_folder(BASE_UPLOAD_DIRECTORY):
    base_dir = Path(BASE_UPLOAD_DIRECTORY)
    hash_value = uuid.uuid4().hex
    hash_folder = base_dir / hash_value
    images_folder = hash_folder / "images_to_process"
    
    hash_folder.mkdir(exist_ok=True)
    images_folder.mkdir(exist_ok=True)
    
    return str(hash_folder), str(images_folder)

def encode_images_base64(image_paths: List[str]) -> List[str]:
    encoded_images = []
    try:
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded_string)
        
        return encoded_images
    except Exception as e:
        raise Exception(f"Error encoding images to base64: {str(e)}")
    

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        raise Exception(f"Error encoding image to base64: {str(e)}")
    

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    return BytesIO(image_data)

def convert_pdf_to_images(pdf_file_path):
    try:
        images = convert_from_path(pdf_file_path, dpi=300)
        return images
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {str(e)}")

def extract_images_and_names_from_zip(zip_file):
    images = []
    image_names = []

    with zipfile.ZipFile(zip_file) as z:
        print("read zip file")
        for filename in z.namelist():
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                with z.open(filename) as img_file:
                    print("proccessing image")
                    image_bytes = BytesIO(img_file.read())
                    image = Image.open(image_bytes)
                    images.append(image)
                    image_names.append(filename)

    print("zip file extraction complete!")
    return images, image_names


def display_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image.show()

def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return images
