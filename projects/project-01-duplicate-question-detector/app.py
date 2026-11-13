"""
Official Project 1: Duplicate Question Detection Implementation
Reference Manual: projects/guides/Project_1_Duplicate_Question_Detection_Learning_Notes.pdf
Architecture: Lightweight sentence-transformers/all-MiniLM-L6-v2 (~80 MB, 384 dimensions)
Strict Zero-Emoji Policy.

Supports:
1. Automated verification and benchmark demonstration
2. Interactive command-line loop (when executed with --interactive or in interactive terminal)
3. FastAPI web application (uvicorn app:app --port 8000)
"""

import os
import sys
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.70

# Official classroom reference questions (from Page 7 of Teaching Manual)
EXISTING_QUESTIONS = {
    1: "How to sort a list in Python?",
    2: "What is the difference between list.sort() and sorted() in Python?",
    3: "How to concatenate strings in Python?",
    4: "How can I merge multiple lists in Python?",
    5: "How do I install a Python package using pip?",
    6: "What is a dictionary in Python and how is it used?",
    7: "How can I read a CSV file using pandas?",
    8: "How do I create a function in Python?",
}


def load_model():
    """Load the sentence-embedding model once."""
    print(f"Loading sentence embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def build_reference_embeddings(model, questions):
    """Encode all reference questions in one batch."""
    question_ids = list(questions.keys())
    question_texts = [questions[q_id] for q_id in question_ids]
    embeddings = model.encode(
        question_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return question_ids, embeddings


def find_duplicates(model, question_ids, reference_embeddings, new_question,
                    threshold=DEFAULT_THRESHOLD, top_k=5):
    """Return reference questions whose cosine similarity meets the threshold."""
    if not new_question or not new_question.strip():
        raise ValueError("Please enter a question before searching.")

    query_embedding = model.encode(
        new_question.strip(),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    similarities = util.cos_sim(query_embedding, reference_embeddings)[0]
    ranked_indices = similarities.argsort(descending=True).tolist()

    matches = []
    for index in ranked_indices:
        score = float(similarities[index].item())
        if score >= threshold:
            q_id = question_ids[index]
            matches.append({
                "question_id": q_id,
                "question_text": EXISTING_QUESTIONS[q_id],
                "similarity": round(score, 4),
            })
        if len(matches) >= top_k:
            break
    return matches


# FastAPI Application Definition (per Section 10 of Teaching Manual)
if HAS_FASTAPI:
    app = FastAPI(
        title="Duplicate Question Detector API",
        description="Lightweight semantic search microservice using all-MiniLM-L6-v2",
        version="1.0"
    )

    class QueryRequest(BaseModel):
        question: str = Field(..., json_schema_extra={"example": "How do I sort a list alphabetically in Python?"})
        threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class DuplicateMatch(BaseModel):
        question_id: int
        question_text: str
        similarity: float

    class QueryResponse(BaseModel):
        query: str
        is_duplicate: bool
        threshold: float
        match_count: int
        matches: List[DuplicateMatch]

    # Pre-load model and embeddings for API service
    _api_model = None
    _api_q_ids = None
    _api_ref_emb = None

    def get_api_resources():
        global _api_model, _api_q_ids, _api_ref_emb
        if _api_model is None:
            _api_model = load_model()
            _api_q_ids, _api_ref_emb = build_reference_embeddings(_api_model, EXISTING_QUESTIONS)
        return _api_model, _api_q_ids, _api_ref_emb

    @app.get("/")
    def root():
        return {
            "status": "online",
            "service": "Duplicate Question Detector",
            "model": MODEL_NAME,
            "reference_questions_count": len(EXISTING_QUESTIONS)
        }

    @app.post("/api/v1/detect-duplicate", response_model=QueryResponse)
    def detect_duplicate(req: QueryRequest):
        m, q_ids, ref_emb = get_api_resources()
        try:
            results = find_duplicates(m, q_ids, ref_emb, req.question, req.threshold, req.top_k)
            return QueryResponse(
                query=req.question,
                is_duplicate=len(results) > 0,
                threshold=req.threshold,
                match_count=len(results),
                matches=[DuplicateMatch(**item) for item in results]
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))


def run_automated_demo(model, question_ids, reference_embeddings):
    """Run standard test cases per Page 11 of Teaching Manual."""
    test_cases = [
        ("Direct Paraphrase", "How do I arrange a Python list alphabetically?"),
        ("Abbreviated query", "How to join two strings together in Python?"),
        ("Related concept", "When should I use sorted() instead of list.sort()?"),
        ("Unrelated subject", "What is photosynthesis?"),
    ]

    print("=" * 80)
    print(f"AUTOMATED VERIFICATION BENCHMARK (Threshold = {DEFAULT_THRESHOLD})")
    print("=" * 80)

    for category, query in test_cases:
        matches = find_duplicates(model, question_ids, reference_embeddings, query, threshold=DEFAULT_THRESHOLD)
        print(f"\n[{category}] Query: '{query}'")
        if matches:
            print(f"  Status: DUPLICATE DETECTED ({len(matches)} match{'es' if len(matches) > 1 else ''})")
            for m in matches:
                print(f"    - ID {m['question_id']} | Similarity: {m['similarity']:.4f} | {m['question_text']}")
        else:
            print("  Status: UNIQUE (No reference question met threshold)")
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE: all-MiniLM-L6-v2 pipeline is fully operational.")
    print("=" * 80)


def run_interactive_loop(model, question_ids, reference_embeddings):
    """Interactive command-line loop per Page 7-8 of Teaching Manual."""
    print("\nInteractive Duplicate Question Detector CLI")
    print(f"Number of reference questions indexed: {len(question_ids)}")
    print("Type a question to detect duplicates. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Enter a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Duplicate Question Detector.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Duplicate Question Detector closed.")
            break
        if not query:
            print("Please enter a non-empty question.\n")
            continue

        try:
            matches = find_duplicates(model, question_ids, reference_embeddings, query, threshold=DEFAULT_THRESHOLD)
            if matches:
                print("\nPossible duplicate / similar questions:")
                for match in matches:
                    print(f"  ID {match['question_id']} | Similarity: {match['similarity']:.4f} | {match['question_text']}")
            else:
                print("\nNo question reached the similarity threshold.")
            print()
        except ValueError as exc:
            print(f"Error: {exc}\n")


def main():
    parser = argparse.ArgumentParser(description="Duplicate Question Detector CLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI loop")
    parser.add_argument("--query", type=str, help="Search for duplicates of a single question")
    args = parser.parse_args()

    model = load_model()
    question_ids, reference_embeddings = build_reference_embeddings(model, EXISTING_QUESTIONS)

    if args.query:
        matches = find_duplicates(model, question_ids, reference_embeddings, args.query, threshold=DEFAULT_THRESHOLD)
        print(f"\nQuery: '{args.query}'")
        if matches:
            for m in matches:
                print(f"  ID {m['question_id']} | Similarity: {m['similarity']:.4f} | {m['question_text']}")
        else:
            print("No duplicates detected above threshold.")
    elif args.interactive or (sys.stdin.isatty() and len(sys.argv) == 1):
        run_interactive_loop(model, question_ids, reference_embeddings)
    else:
        run_automated_demo(model, question_ids, reference_embeddings)


if __name__ == "__main__":
    main()
