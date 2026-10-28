import json
import os
import pickle
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks.json")
INDEX_FILE = os.path.join(BASE_DIR, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(BASE_DIR, "data", "metadata.json")
BM25_FILE = os.path.join(BASE_DIR, "data", "bm25.pkl")


# Load embedding model
model = SentenceTransformer("BAAI/bge-large-en")

def build_index():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["embedding_text"] for c in chunks]

    print("🔄 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (with normalized embeddings)
    index.add(np.array(embeddings))

    # Save index
    faiss.write_index(index, INDEX_FILE)

    # Save metadata separately
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 🔹 BM25 (NEW PART)
    # -------------------------
    print("🔄 Building BM25 index...")

    tokenized_corpus = [
        c["embedding_text"].lower().split()
        for c in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25
    with open(BM25_FILE, "wb") as f:
        pickle.dump((bm25, tokenized_corpus), f)

    print("✅ FAISS + BM25 index built and saved!")

if __name__ == "__main__":
    build_index()