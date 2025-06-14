from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from embeddings import get_embedding, cosine_similarity

app = FastAPI()

# Load your course data and precomputed embeddings here
DOCUMENTS = [...]     # List of texts
EMBEDDINGS = np.load("data/embeddings.npy")

class Query(BaseModel):
    question: str

@app.post("/answer")
async def answer_question(query: Query):
    query_embedding = get_embedding(query.question)
    scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in EMBEDDINGS]
    best_idx = np.argmax(scores)
    return {"answer": DOCUMENTS[best_idx]}
