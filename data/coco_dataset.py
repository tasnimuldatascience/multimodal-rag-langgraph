import os
from datasets import load_dataset
from PIL import Image
from embeddings.multimodal import embed_image
from utils.vector_store import add_to_db

def populate_coco(n_samples=10):
    custom_cache_dir = "./data/coco_cache"
    coco = load_dataset(
        "RIW/small-coco",
        split=f"train[:{n_samples}]",
        cache_dir=custom_cache_dir
    )
    
    os.makedirs('data/images', exist_ok=True)

    for i, item in enumerate(coco):
        image = item['image']
        image_path = f"data/images/{i}.jpg"
        image.save(image_path)

        embedding = embed_image(image_path)

        # Handle captions (annotations) carefully:
        annotations = item.get('annotations', [])
        caption = annotations[0]['caption'] if annotations and 'caption' in annotations[0] else 'No caption'

        add_to_db(embedding, image_path, 'image', {"caption": caption})
        print(f"✅ Added image {image_path}")
