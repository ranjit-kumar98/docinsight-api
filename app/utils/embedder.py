from sentence_transformers import SentenceTransformer

# Load once (keep global)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text).tolist()

def embed_chunks(chunks):
    embedded = []
    for chunk in chunks:
        vector = embed_text(chunk)
        embedded.append({
            "chunk": chunk,
            "embedding": vector
        })
    return embedded

def embed_text(text):
    return model.encode(text).tolist()