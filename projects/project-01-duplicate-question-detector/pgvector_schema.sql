-- Optional database extension exercise for PostgreSQL with pgvector installed.
-- all-MiniLM-L6-v2 produces 384-dimensional embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS questions (
    question_id BIGSERIAL PRIMARY KEY,
    question_text TEXT NOT NULL UNIQUE,
    embedding VECTOR(384) NOT NULL
);

-- Optional approximate-nearest-neighbour index (requires a pgvector version
-- that supports HNSW indexes).
CREATE INDEX IF NOT EXISTS questions_embedding_hnsw_idx
ON questions USING hnsw (embedding vector_cosine_ops);

-- Example nearest-neighbour query. Bind the input vector as a parameter from
-- Python rather than typing the placeholder literally in pgAdmin.
-- SELECT question_id, question_text,
--        1 - (embedding <=> '[...]'::vector) AS cosine_similarity
-- FROM questions
-- ORDER BY embedding <=> '[...]'::vector
-- LIMIT 5;
