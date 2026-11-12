"""
Experiment 10: Discourse Analysis and Coherence Modeling
Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools
Reference: notes/day-03/code/Day 3/discourse_analysis.py

Key Objectives:
1. Identify explicit rhetorical discourse markers (connectives) and classify their semantic relations:
   - Causal / Result (therefore, because, as a result)
   - Contrastive / Adversative (however, although, but)
   - Additive / Elaboration (moreover, furthermore, in addition)
2. Compute inter-sentential semantic similarity using dense neural embeddings (all-MiniLM-L6-v2)
   with fallback to lexical TF-IDF cosine similarity.
3. Quantify consecutive discourse coherence metrics across paragraph spans.
Strict Zero-Emoji Policy.
"""

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DISCOURSE_RELATIONS = {
    "therefore": "Causal / Consequence",
    "as a result": "Causal / Consequence",
    "because": "Causal / Explanation",
    "consequently": "Causal / Consequence",
    "however": "Adversative / Contrast",
    "although": "Adversative / Concession",
    "but": "Adversative / Contrast",
    "moreover": "Additive / Elaboration",
    "furthermore": "Additive / Elaboration",
    "in addition": "Additive / Elaboration"
}


def load_embedding_pipeline():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda sents: model.encode(sents)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer()
        return lambda sents: vec.fit_transform(sents).toarray()


def analyze_discourse(text):
    sentences = sent_tokenize(text.strip())
    encode_func = load_embedding_pipeline()
    embeddings = encode_func(sentences)
    sim_matrix = cosine_similarity(embeddings)

    print("=" * 80)
    print("EXPERIMENT 10: DISCOURSE ANALYSIS & COHERENCE MODELING")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools")
    print("=" * 80)

    print(f"\nDiscourse Text ({len(sentences)} sentences):")
    for i, s in enumerate(sentences, 1):
        print(f"  [{i}] {s}")

    print("\n" + "=" * 80)
    print("1. DISCOURSE MARKER & RHETORICAL RELATION IDENTIFICATION")
    print("=" * 80)
    for i, s in enumerate(sentences, 1):
        s_lower = s.lower()
        found = [(m, DISCOURSE_RELATIONS[m]) for m in DISCOURSE_RELATIONS if m in s_lower]
        if found:
            relations_str = ", ".join([f"'{m}' ({rel})" for m, rel in found])
            print(f"Sentence {i}: Connectives -> {relations_str}")
        else:
            print(f"Sentence {i}: Implicit coherence (no explicit surface connective)")

    print("\n" + "=" * 80)
    print("2. CONSECUTIVE SENTENTIAL COHERENCE SCORES")
    print("=" * 80)
    coherence_scores = []
    for i in range(len(sentences) - 1):
        score = sim_matrix[i][i + 1]
        coherence_scores.append(score)
        print(f"Transition [{i} -> {i+1}]: Similarity = {score:.4f}")
        print(f"  S{i}  : \"{sentences[i]}\"")
        print(f"  S{i+1}: \"{sentences[i+1]}\"")

    mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    print("-" * 80)
    print(f"Mean Global Consecutive Coherence Score: {mean_coherence:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    discourse_text = """
    Ravi prepared diligently for his academic evaluations.
    Therefore, he attained exceptional scholastic marks.
    However, he found computational complexity theory demanding.
    Moreover, he solved supplementary algorithmic exercises.
    As a result, his analytical problem-solving capability flourished.
    """
    analyze_discourse(discourse_text)

