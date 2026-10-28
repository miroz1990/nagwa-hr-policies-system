import os
import json
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_DIR, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(BASE_DIR, "data", "metadata.json")
BM25_FILE = os.path.join(BASE_DIR, "data", "bm25.pkl")

embedder = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def load_bm25():
    try:
        with open(BM25_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise Exception("BM25 file not found. Rebuild the index/BM25 file first.")


bm25, tokenized_corpus = load_bm25()


def normalize_query(query):
    query = query.lower()
    query = query.replace("nagwa", "")
    query = query.replace("in the company", "")
    query = query.replace("at the company", "")
    query = query.replace("company policy", "")
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def section_boost_score(query, section, text):
    query_tokens = set(tokenize(normalize_query(query)))
    section_tokens = set(tokenize(section))
    text_tokens = set(tokenize(text))

    boost = 0.0

    # Strong boost when query topic appears in section title
    if query_tokens & section_tokens:
        boost += 2.0

    # Medium boost when query topic appears in chunk text
    if query_tokens & text_tokens:
        boost += 0.5

    # Manual topic boosts for common HR sections
    q = normalize_query(query)
    section_lower = section.lower()

    topic_map = {
        "smoking": ["smoking"],
        "internet": ["internet"],
        "mobile": ["mobile phones", "mobile"],
        "phone": ["mobile phones", "mobile"],
        "headphone": ["music players & headphones", "headphones"],
        "music": ["music players & headphones", "music"],
        "prayer": ["prayer times", "prayer"],
        "praying": ["prayer times", "prayer"],
        "pray": ["prayer times", "prayer"],
        "overtime": ["overtime"],
        "annual leave": ["annual leave", "annual leaves"],
        "sick leave": ["sick leave"],
        "maternity": ["maternity leave"],
        "paternity": ["paternity leave"],
        "part time": ["part-time", "part time"],
        "part-timer": ["part-time", "part time"],
        "insurance": ["medical", "health insurance"],
        "medical": ["medical", "health insurance"],
        "working hours": ["working hours", "flexible working hours"],
    }

    for query_term, section_terms in topic_map.items():
        if query_term in q:
            for term in section_terms:
                if term in section_lower:
                    boost += 20.0

    return boost


def retrieve(query, top_k=40, rerank_k=10):
    retrieval_query = normalize_query(query)

    # -------------------------
    # Hard topic routing
    # -------------------------
    if any(word in retrieval_query for word in ["prayer", "praying", "pray"]):
        prayer_chunks = [
            c for c in metadata
            if "prayer" in c.get("section", "").lower()
        ]

        if prayer_chunks:
            return [
                {
                    "text": c.get("text", ""),
                    "section": c.get("section", "Unknown"),
                    "document": c.get("document", "Unknown"),
                    "source": c.get("source", "Unknown"),
                    "page": c.get("page"),
                    "type": c.get("type", "paragraph"),
                    "hybrid_score": 1.0,
                    "rerank_score": 1.0,
                    "section_boost": 100.0,
                    "final_score": 100.0,
                }
                for c in prayer_chunks[:rerank_k]
            ]

    # -------------------------
    # BM25 scores
    # -------------------------
    tokenized_query = retrieval_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # -------------------------
    # FAISS search
    # -------------------------
    query_embedding = embedder.encode([retrieval_query], normalize_embeddings=True)
    faiss_scores, faiss_indices = index.search(np.array(query_embedding), top_k)

    embedding_scores = np.zeros(len(metadata))
    for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        embedding_scores[idx] = score

    # -------------------------
    # Normalize scores
    # -------------------------
    bm25_scores = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-8
    )

    embedding_scores = (embedding_scores - embedding_scores.min()) / (
        embedding_scores.max() - embedding_scores.min() + 1e-8
    )

    # -------------------------
    # Hybrid scoring
    # -------------------------
    alpha = 0.65
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * embedding_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        chunk = metadata[idx]
        candidates.append({
            "text": chunk.get("text", ""),
            "section": chunk.get("section", "Unknown"),
            "document": chunk.get("document", "Unknown"),
            "source": chunk.get("source", "Unknown"),
            "page": chunk.get("page"),
            "type": chunk.get("type", "paragraph"),
            "hybrid_score": float(hybrid_scores[idx]),
        })

    # -------------------------
    # Cross-encoder reranking
    # Include metadata in reranker input
    # -------------------------
    pairs = [
        (
            retrieval_query,
            f"Document: {c['document']}\nSection: {c['section']}\nText: {c['text']}"
        )
        for c in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        boost = section_boost_score(query, c["section"], c["text"])
        c["rerank_score"] = float(rerank_scores[i])
        c["section_boost"] = boost
        c["final_score"] = c["rerank_score"] + boost

    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    q = normalize_query(query)

    if any(word in q for word in ["prayer", "praying", "pray"]):
        prayer_chunks = [
            c for c in candidates
            if "prayer" in c["section"].lower()
        ]

        if prayer_chunks:
            prayer_chunks = sorted(
                prayer_chunks,
                key=lambda x: x["final_score"],
                reverse=True
            )

            remaining = [
                c for c in candidates
                if "prayer" not in c["section"].lower()
            ]

            return prayer_chunks + remaining[:rerank_k-1]

    return candidates[:rerank_k]


if __name__ == "__main__":
    query = "Is smoking allowed in Nagwa?"
    results = retrieve(query)

    for r in results:
        print("\n---")
        print("Document:", r["document"])
        print("Section:", r["section"])
        print("Text:", r["text"])
        print("Hybrid:", r["hybrid_score"])
        print("Rerank:", r["rerank_score"])
        print("Boost:", r["section_boost"])
        print("Final:", r["final_score"])