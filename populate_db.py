from data.wikipedia_dataset import populate_wikipedia
from data.coco_dataset import populate_coco

topics = [
    'Generative artificial intelligence', 
    'Artificial intelligence', 
    'Deep learning', 
    'Generative adversarial network'
]

populate_wikipedia(topics)
populate_coco(n_samples=10)
