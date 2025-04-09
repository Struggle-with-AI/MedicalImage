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