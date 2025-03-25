from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def embed_text(text, max_length=77):
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length  # explicitly set maximum length
    )
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs).numpy()[0]
    return embedding / np.linalg.norm(embedding)

def embed_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).numpy()[0]
    return embedding / np.linalg.norm(embedding)
