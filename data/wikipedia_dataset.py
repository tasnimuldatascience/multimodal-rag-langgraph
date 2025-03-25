import wikipedia
from embeddings.multimodal import embed_text
from utils.vector_store import add_to_db

def populate_wikipedia(topics):
    for topic in topics:
        try:
            # Explicitly specify auto_suggest=False to prevent bad suggestions
            page = wikipedia.page(topic, auto_suggest=False)
            embedding = embed_text(page.summary)
            add_to_db(embedding, page.summary, 'text', {"title": page.title})
            print(f"✅ Added page: {page.title}")
        except wikipedia.exceptions.DisambiguationError as e:
            # resolve first suggested page from disambiguation
            page = wikipedia.page(e.options[0])
            embedding = embed_text(page.summary)
            add_to_db(embedding, page.summary, 'text', {"title": page.title})
            print(f"⚠️ Disambiguation resolved to: {page.title}")
        except wikipedia.exceptions.PageError as e:
            print(f"❌ PageError for {topic}: {e}")
