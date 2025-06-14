# generate_embeddings.py

import json
import numpy as np
from embeddings import get_embedding
import os

# Load your documents
def load_documents():
    transcripts = json.load(open("content/transcripts.json", "r", encoding="utf-8"))
    posts = json.load(open("content/posts.json", "r", encoding="utf-8"))

    # Example: if each item has a "text" field
    transcript_texts = [item["text"] for item in transcripts]
    post_texts = [item["text"] for item in posts]

    return transcript_texts + post_texts

def main():
    documents = load_documents()
    embeddings = []

    print(f"Generating embeddings for {len(documents)} documents...")

    for i, doc in enumerate(documents):
        try:
            embedding = get_embedding(doc)
            embeddings.append(embedding)
            print(f"[{i+1}/{len(documents)}] Embedded successfully.")
        except Exception as e:
            print(f"Error at doc {i}: {e}")
            embeddings.append(np.zeros(1536))  # fallback if needed

    embeddings = np.array(embeddings)
    
    os.makedirs("data", exist_ok=True)
    np.save("data/embeddings.npy", embeddings)
    print("Embeddings saved to data/embeddings.npy")

if __name__ == "__main__":
    main()
