"""Optional FastAPI interface for the duplicate-question detector.

Run from this folder with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUESTIONS = {
    1: "How to sort a list in Python?",
    2: "What is the difference between list.sort() and sorted() in Python?",
    3: "How to concatenate strings in Python?",
    4: "How can I merge multiple lists in Python?",
    5: "How do I install a Python package using pip?",
    6: "What is a dictionary in Python and how is it used?",
    7: "How can I read a CSV file using pandas?",
    8: "How do I create a function in Python?",
}

app = FastAPI(title="Duplicate Question Detector", version="1.0")
model = SentenceTransformer(MODEL_NAME)
question_ids = list(QUESTIONS.keys())
reference_embeddings = model.encode(
    [QUESTIONS[q_id] for q_id in question_ids],
    convert_to_tensor=True,
    normalize_embeddings=True,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, description="Question to check")
    threshold: float = Field(default=0.70, ge=-1.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def home():
    return {"message": "Duplicate Question Detector API is running."}


@app.post("/detect")
def detect_duplicates(request: QueryRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    query_embedding = model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    scores = util.cos_sim(query_embedding, reference_embeddings)[0]
    order = scores.argsort(descending=True).tolist()
    results = []
    for index in order:
        score = float(scores[index].item())
        if score >= request.threshold:
            q_id = question_ids[index]
            results.append({
                "question_id": q_id,
                "question_text": QUESTIONS[q_id],
                "similarity": round(score, 4),
            })
        if len(results) >= request.top_k:
            break
    return {"input_question": query, "threshold": request.threshold,
            "matches": results, "match_found": bool(results)}
