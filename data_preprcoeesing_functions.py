import os
import re
from PIL import Image
import numpy as np

def get_images_by_folder_id(folder_id):
    """
    Given an image folder ID (e.g., '1.3.6.1.4.1...'),
    finds all images in that folder matching the pattern '1-xxx.jpg'.

    Parameters:
        folder_id (str): Name of the image folder

    Returns:
        List of (filename, image array) tuples
    """
    # Build full image path from folder_id
    image_file_path = os.path.join(
        "kagglehub",
        "datasets",
        "awsaf49",
        "cbis-ddsm-breast-cancer-image-dataset",
        "versions",
        "1",
        "jpeg",
        folder_id
    )

    pattern = re.compile(r"^1-\d{3}\.jpg$", re.IGNORECASE)
    matched_images = []

    for file in os.listdir(image_file_path):
        if pattern.match(file):
            full_path = os.path.join(image_file_path, file)
            img = Image.open(full_path).convert("RGB")
            matched_images.append((file, np.array(img)))

    return matched_images

def extract_folder_id(image_file_path: str) -> str:
    parts = image_file_path.split('/')
    if len(parts) >= 3:
        return parts[2]
    return None

def merge_benign_label(df):
    df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
    return df

def get_image_by_uid_and_filename(folder_uid, file_name, base_jpeg_path):
    """
    Given a folder UID and image filename, return the image as a PIL.Image.

    Parameters:
        folder_uid (str): The UID of the folder where the image is stored.
        file_name (str): The exact image filename (e.g., '1-188.jpg').
        base_jpeg_path (str): Path to the root jpeg directory.

    Returns:
        Tuple[str, PIL.Image] if found, else None
    """
    # Normalize filename
    file_name = file_name.lower().strip()

    # Build full image path
    image_path = os.path.join(base_jpeg_path, folder_uid, file_name)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    # Load and return the image
    image = Image.open(image_path).convert("RGB")
    return file_name, image
