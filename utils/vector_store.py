import chromadb
import uuid
from chromadb.errors import InvalidCollectionException

# explicitly use persistent storage
client = chromadb.PersistentClient(path="./.chroma")

try:
    client.get_collection(name="multimodal_rag")
    client.delete_collection(name="multimodal_rag")
    print("✅ Deleted existing collection.")
except InvalidCollectionException:
    print("⚠️ Collection didn't exist, no need to delete.")

collection = client.create_collection(name="multimodal_rag")
print("✅ Created new collection 'multimodal_rag'.")

def add_to_db(embedding, content, modality, metadata=None):
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[content],
        metadatas=[{"modality": modality, **(metadata or {})}],
        ids=[str(uuid.uuid4())]
    )

def query_db(query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results
