from embeddings.multimodal import embed_text, embed_image
from utils.vector_store import query_db

def retrieval_agent(input_data):
    query = input_data['query']
    modality = input_data['modality']

    embedding = embed_text(query) if modality == 'text' else embed_image(query)
    retrieved = query_db(embedding)

    # Explicitly confirm documents retrieved
    if retrieved['documents']:
        input_data['context'] = "\n---\n".join(retrieved['documents'][0])
    else:
        input_data['context'] = "No context found."

    print("Retrieved Context:", input_data['context'])  # debug print clearly
    return input_data
