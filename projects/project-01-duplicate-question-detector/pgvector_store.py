"""Optional PostgreSQL + pgvector persistence for MiniLM question embeddings.

Prerequisites:
  1. PostgreSQL server with the pgvector extension installed.
  2. pip install psycopg2-binary pgvector
  3. Set DATABASE_URL in PowerShell, for example:
     $env:DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/nlpdb"

Run from this folder:
  python pgvector_store.py
"""

import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_QUESTIONS = [
    "How to sort a list in Python?",
    "What is the difference between list.sort() and sorted() in Python?",
    "How to concatenate strings in Python?",
    "How can I merge multiple lists in Python?",
]


def connect_database():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before running this optional database exercise.")
    connection = psycopg2.connect(database_url)
    register_vector(connection)
    return connection


def initialize_table(connection):
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                question_id BIGSERIAL PRIMARY KEY,
                question_text TEXT NOT NULL UNIQUE,
                embedding VECTOR(384) NOT NULL
            );
        """)
    connection.commit()


def store_questions(connection, model, questions):
    for question in questions:
        vector = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        vector = np.asarray(vector, dtype=np.float32)
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO questions (question_text, embedding)
                   VALUES (%s, %s)
                   ON CONFLICT (question_text)
                   DO UPDATE SET embedding = EXCLUDED.embedding;""",
                (question, vector),
            )
    connection.commit()


def search_similar_questions(connection, model, query, top_k=5):
    query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT question_id, question_text,
                      1 - (embedding <=> %s) AS similarity
               FROM questions
               ORDER BY embedding <=> %s
               LIMIT %s;""",
            (query_vector, query_vector, top_k),
        )
        return cursor.fetchall()


def main():
    print("Loading MiniLM sentence-embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    connection = connect_database()
    try:
        initialize_table(connection)
        store_questions(connection, model, SAMPLE_QUESTIONS)
        query = input("Enter a question to search in PostgreSQL (or press Enter for a demo query): ").strip()
        if not query:
            query = "How do I arrange a Python list?"
        print("\nNearest questions from pgvector:")
        for question_id, question_text, similarity in search_similar_questions(connection, model, query):
            print(f"ID {question_id} | similarity={float(similarity):.4f} | {question_text}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
