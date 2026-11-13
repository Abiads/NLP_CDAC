"""PostgreSQL + pgvector persistence for MiniLM question embeddings.

Features:
  1. Real PostgreSQL + pgvector mode when DATABASE_URL is set and reachable.
     Uses VECTOR(384) column and cosine distance operator (<=>).
  2. In-memory SQLite vector emulation mode when PostgreSQL is offline or
     DATABASE_URL is not configured.
  3. Supports interactive queries, CLI arguments (--query, --demo), and
     custom top-k thresholds.

Prerequisites for real PostgreSQL mode:
  1. PostgreSQL server with the pgvector extension installed.
  2. pip install psycopg2-binary pgvector
  3. Set DATABASE_URL in PowerShell, for example:
     $env:DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/nlpdb"

Usage:
  python pgvector_store.py
  python pgvector_store.py --demo
  python pgvector_store.py --query "How do I sort a list in Python?"
"""

import argparse
import os
import sqlite3
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports for real PostgreSQL + pgvector
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    register_vector = None
    PSYCOPG2_AVAILABLE = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_QUESTIONS = [
    "How to sort a list in Python?",
    "What is the difference between list.sort() and sorted() in Python?",
    "How to concatenate strings in Python?",
    "How can I merge multiple lists in Python?",
    "How do I install a Python package using pip?",
    "What is a dictionary in Python and how is it used?",
    "How can I read a CSV file using pandas?",
    "How do I create a function in Python?",
]


# ==============================================================================
# In-Memory SQLite Vector Store (Emulation Mode)
# ==============================================================================

class SqliteVectorStore:
    """In-memory SQLite vector store emulating pgvector behavior."""

    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self._initialize_table()

    def _initialize_table(self):
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_text TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL
                );
            """)

    def store_questions(self, model: SentenceTransformer, questions: List[str]):
        embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
        with self.connection:
            for text, vector in zip(questions, embeddings):
                vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
                self.connection.execute(
                    """
                    INSERT INTO questions (question_text, embedding)
                    VALUES (?, ?)
                    ON CONFLICT(question_text) DO UPDATE SET embedding = excluded.embedding;
                    """,
                    (text, vector_bytes),
                )

    def search_similar_questions(
        self, model: SentenceTransformer, query: str, top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = np.asarray(query_vector, dtype=np.float32)

        cursor = self.connection.cursor()
        cursor.execute("SELECT question_id, question_text, embedding FROM questions")
        rows = cursor.fetchall()

        results = []
        for q_id, q_text, emb_bytes in rows:
            stored_vector = np.frombuffer(emb_bytes, dtype=np.float32)
            # Dot product equals cosine similarity for normalized vectors
            cosine_similarity = float(np.dot(query_vector, stored_vector))
            results.append((q_id, q_text, cosine_similarity))

        # Sort by similarity descending (equivalent to ORDER BY embedding <=> query_vector ASC)
        results.sort(key=lambda item: item[2], reverse=True)
        return results[:top_k]

    def close(self):
        self.connection.close()


# ==============================================================================
# PostgreSQL + pgvector Store (Production Mode)
# ==============================================================================

class PgVectorStore:
    """PostgreSQL vector store using the pgvector extension."""

    def __init__(self, database_url: str):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "psycopg2 or pgvector Python package is missing. "
                "Install with: pip install psycopg2-binary pgvector"
            )
        self.connection = psycopg2.connect(database_url)
        register_vector(self.connection)

    def initialize_table(self):
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    question_id BIGSERIAL PRIMARY KEY,
                    question_text TEXT NOT NULL UNIQUE,
                    embedding VECTOR(384) NOT NULL
                );
            """)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS questions_embedding_hnsw_idx
                    ON questions USING hnsw (embedding vector_cosine_ops);
                """)
            except Exception:
                # Older pgvector versions may not support HNSW indexing
                self.connection.rollback()
        self.connection.commit()

    def store_questions(self, model: SentenceTransformer, questions: List[str]):
        for question in questions:
            vector = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
            vector = np.asarray(vector, dtype=np.float32)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO questions (question_text, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (question_text)
                    DO UPDATE SET embedding = EXCLUDED.embedding;
                    """,
                    (question, vector),
                )
        self.connection.commit()

    def search_similar_questions(
        self, model: SentenceTransformer, query: str, top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = np.asarray(query_vector, dtype=np.float32)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT question_id, question_text,
                       1 - (embedding <=> %s) AS similarity
                FROM questions
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_vector, query_vector, top_k),
            )
            return cursor.fetchall()

    def close(self):
        self.connection.close()


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_vector_store(database_url: str = None):
    """Factory returning PgVectorStore if available, otherwise SqliteVectorStore."""
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url and PSYCOPG2_AVAILABLE:
        try:
            store = PgVectorStore(db_url)
            store.initialize_table()
            print("[Backend] Connected to PostgreSQL with pgvector extension.")
            return store
        except Exception as error:
            print(f"[Notice] Failed to connect to PostgreSQL ({error}).")
            print("[Notice] Falling back to SQLite vector emulation mode.")
    else:
        if not PSYCOPG2_AVAILABLE:
            print("[Notice] psycopg2 or pgvector package not installed.")
        elif not db_url:
            print("[Notice] DATABASE_URL is not set.")
        print("[Notice] Using in-memory SQLite vector store emulation mode.")
        print("To use real PostgreSQL + pgvector:")
        print("  1. pip install psycopg2-binary pgvector")
        print("  2. $env:DATABASE_URL = 'postgresql://postgres:password@localhost:5432/nlpdb'")
        print("  3. Ensure pgvector extension is installed on PostgreSQL.")

    store = SqliteVectorStore()
    return store


def run_search_demo(store, model: SentenceTransformer, query: str, top_k: int = 5, threshold: float = 0.70):
    """Execute vector search and display ranked results."""
    print(f"\nQuery: '{query}'")
    print(f"Top-K: {top_k} | Duplicate Threshold: {threshold:.2f}")
    print("-" * 75)

    results = store.search_similar_questions(model, query, top_k=top_k)
    if not results:
        print("No matching questions found.")
        return

    matches_found = 0
    for rank, (question_id, question_text, similarity) in enumerate(results, start=1):
        is_duplicate = float(similarity) >= threshold
        status = "[DUPLICATE MATCH]" if is_duplicate else "[BELOW THRESHOLD]"
        if is_duplicate:
            matches_found += 1
        print(f"  #{rank} ID {question_id:2d} | Similarity: {float(similarity):.4f} | {status}")
        print(f"      Question: {question_text}")

    print("-" * 75)
    print(f"Total candidate matches above {threshold:.2f}: {matches_found}/{len(results)}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pgvector question store and similarity search for Duplicate Question Detector."
    )
    parser.add_argument("--database-url", type=str, default=None, help="PostgreSQL connection string")
    parser.add_argument("--query", type=str, default=None, help="Question to search against stored database")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--threshold", type=float, default=0.70, help="Duplicate similarity threshold")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration queries")
    args = parser.parse_args()

    print("=" * 75)
    print("Duplicate Question Detector - pgvector Store")
    print("Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)")
    print("=" * 75)

    print("Loading MiniLM sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    store = get_vector_store(args.database_url)

    try:
        print(f"Indexing {len(SAMPLE_QUESTIONS)} reference questions into vector store...")
        store.store_questions(model, SAMPLE_QUESTIONS)
        print("Indexing completed successfully.")

        if args.demo:
            demo_queries = [
                "How do I arrange a Python list in ascending order?",
                "What is the best way to combine strings in Python?",
                "How do I read a dataset CSV into pandas dataframe?",
                "What is the capital of France?",
            ]
            for query in demo_queries:
                run_search_demo(store, model, query, top_k=args.top_k, threshold=args.threshold)
        elif args.query:
            run_search_demo(store, model, args.query, top_k=args.top_k, threshold=args.threshold)
        else:
            # Interactive mode with fallback for non-TTY or EOF
            try:
                user_query = input(
                    "\nEnter a question to search (or press Enter for demo): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                user_query = ""

            if not user_query:
                user_query = "How do I arrange a Python list in ascending order?"
                print(f"Using default demo query: '{user_query}'")

            run_search_demo(store, model, user_query, top_k=args.top_k, threshold=args.threshold)

    finally:
        store.close()
        print("\nVector store connection closed.")


if __name__ == "__main__":
    main()
