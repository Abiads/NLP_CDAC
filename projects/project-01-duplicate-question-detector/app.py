"""
Environment Verification and Lightweight Duplicate Question Detector CLI
Primary Guide: projects/guides/NLP Environment Setup & Duplicate Question Detector Guide.pdf
Model: sentence-transformers/all-MiniLM-L6-v2 (~80 MB, 384 dimensions)
Strict Zero-Emoji Policy.
"""

import sys
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

def verify_environment():
    print("=" * 80)
    print("NLP ENVIRONMENT VERIFICATION: LIGHTWEIGHT MINILM PIPELINE")
    print("=" * 80)
    print(f"[Loading] SentenceTransformer model: {MODEL_NAME} (Size: ~80 MB)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[Success] Loaded model: {MODEL_NAME}")
    print(f"  - Embedding Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  - Max Sequence Length: {model.max_seq_length}")
    print(f"  - Total Parameters: ~22 Million (< 90 MB memory footprint)")
    print("-" * 80)
    return model

def run_demo(model):
    # In-memory database of existing questions (per Section 5 of guide)
    existing_questions = {
        1: "How to sort a list in Python?",
        2: "What's the difference between list.sort() and sorted()?",
        3: "How to concatenate strings in Python?",
        4: "Merge multiple lists in Python",
        5: "How to convert string to integer in Python?",
        6: "How do I reverse a list or string in Python?",
        7: "What is the difference between shallow copy and deep copy in Python?",
        8: "How to read and parse CSV files with Python?"
    }

    print("\n[Indexing] Encoding reference question database...")
    question_embeddings = {
        q_id: model.encode(text, convert_to_tensor=True)
        for q_id, text in existing_questions.items()
    }
    print(f"[Done] Indexed {len(question_embeddings)} questions.")

    test_queries = [
        "How do I sort a list alphabetically in Python?",
        "Can you explain the difference between sorted() and sort()?",
        "Combine two strings together in Python",
        "How to deploy a Docker container to AWS?"  # Out-of-domain query
    ]

    threshold = 0.70
    print("\n" + "=" * 80)
    print(f"EVALUATING QUERIES (Threshold = {threshold})")
    print("=" * 80)

    for query in test_queries:
        query_emb = model.encode(query, convert_to_tensor=True)
        matches = []
        for q_id, ref_emb in question_embeddings.items():
            sim = util.pytorch_cos_sim(query_emb, ref_emb).item()
            if sim >= threshold:
                matches.append({
                    "id": q_id,
                    "text": existing_questions[q_id],
                    "similarity": round(sim, 4)
                })
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        print(f"\nIncoming Query: '{query}'")
        if matches:
            print(f"  [Status: DUPLICATE DETECTED] Top match found:")
            for m in matches:
                print(f"    - [ID: {m['id']}] '{m['text']}' (Cosine Similarity: {m['similarity']:.4f})")
        else:
            print(f"  [Status: UNIQUE QUESTION] No duplicate exceeding threshold {threshold}.")

if __name__ == "__main__":
    loaded_model = verify_environment()
    run_demo(loaded_model)
