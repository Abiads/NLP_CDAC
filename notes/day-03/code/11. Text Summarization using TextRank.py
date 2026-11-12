"""
Experiment 11: Graph-Based Extractive Text Summarization using TextRank
Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools
Reference: notes/day-03/code/Day 3/text_summarization.py

Key Objectives:
1. Construct an undirected sentence similarity graph from preprocessed token sets.
2. Compute edge weights using Jaccard word-overlap or TF-IDF cosine similarity.
3. Apply the PageRank centrality algorithm to rank sentence salience.
4. Extract the top-k highest-scoring sentences while preserving chronological document order.
Strict Zero-Emoji Policy.
"""

import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


def preprocess_sentence(sentence, stemmer, stop_words):
    tokens = word_tokenize(sentence.lower())
    return [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]


def compute_overlap_similarity(words1, words2):
    s1, s2 = set(words1), set(words2)
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / (len(s1) + len(s2))


def textrank_summarize(document_text, num_sentences=3):
    sentences = sent_tokenize(document_text.strip())
    if len(sentences) <= num_sentences:
        return document_text.strip(), sentences

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    processed = [preprocess_sentence(s, stemmer, stop_words) for s in sentences]

    # Build similarity graph
    graph = nx.Graph()
    for i in range(len(sentences)):
        graph.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            weight = compute_overlap_similarity(processed[i], processed[j])
            if weight > 0:
                graph.add_edge(i, j, weight=weight)

    # PageRank scoring
    scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Top-k selection in original document order
    top_indices = sorted([idx for idx, _ in ranked[:num_sentences]])
    summary = " ".join([sentences[idx] for idx in top_indices])

    return summary, ranked, sentences


def run_benchmark():
    document = """
    Artificial Intelligence is transforming modern industrial paradigms and computational science.
    It is extensively applied across healthcare diagnostic systems, autonomous vehicles, and agriculture.
    Machine learning constitutes a pivotal foundational domain within artificial intelligence.
    It empowers computer architectures to discern predictive representations directly from historical empirical data.
    Deep learning architectures employ multi-layered neural networks to resolve intricate high-dimensional challenges.
    Artificial intelligence aids medical radiologists in detecting abnormalities within diagnostic radiography scans.
    It concurrently underpins adaptive personalized pedagogical curriculum delivery in higher education.
    Modern natural language systems elevate organizational operational productivity and automated document parsing.
    Nonetheless, large-scale artificial intelligence models engender valid ethical, safety, and bias governance questions.
    Global researchers are aggressively formalizing trustworthy, interpretable, and verifiable algorithmic methodologies.
    """

    summary, ranked_scores, sentences = textrank_summarize(document, num_sentences=3)

    print("=" * 80)
    print("EXPERIMENT 11: EXTRACTIVE TEXT SUMMARIZATION (TEXTRANK)")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools")
    print("=" * 80)

    print(f"\nOriginal Source Document ({len(sentences)} sentences):")
    for idx, s in enumerate(sentences, 1):
        print(f"  [{idx:02d}] {s}")

    print("\n" + "=" * 80)
    print("PAGERANK SENTENCE SALIENCE SCORES")
    print("=" * 80)
    for idx, score in ranked_scores:
        marker = "(*)" if idx in [r[0] for r in ranked_scores[:3]] else "   "
        print(f"{marker} S{idx+1:02d} [Centrality: {score:.4f}]: {sentences[idx]}")

    print("\n" + "=" * 80)
    print("EXTRACTED SUMMARY (TOP 3 CENTRAL SENTENCES)")
    print("=" * 80)
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
