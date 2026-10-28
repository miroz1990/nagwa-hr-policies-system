from fastapi import FastAPI
from pydantic import BaseModel

from retriever import retrieve
from llm import generate_answer

app = FastAPI()

# -----------------------
# Request schema
# -----------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------
# RAG core logic
# -----------------------
def normalize_query(query):
    query = query.replace("in Nagwa", "")
    query = query.replace("at Nagwa", "")
    query = query.replace("Nagwa", "")
    return query.strip()

def ask(query):
    clean_query = normalize_query(query)
    contexts = retrieve(clean_query)
    answer = generate_answer(query, contexts)
    return answer, contexts


# -----------------------
# API endpoint
# -----------------------
@app.post("/ask")
def ask_endpoint(req: QueryRequest):
    answer, contexts = ask(req.query)

    return {
        "answer": answer,
        "contexts": contexts
    }