"""Build and execute Assignment 04 Notebook: Statistical Approaches in NLP (Experiments 1-11)."""

import json
import os
import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = r"assignments\assignment-04\assignment_04.ipynb"

nb = nbformat.v4.new_notebook()

cells = [
    nbformat.v4.new_markdown_cell(
        "# Day 4 Assignment: Statistical Approaches in NLP\n\n"
        "**Course:** Natural Language Processing & Computer Vision (CDAC)\n"
        "**Curriculum Reference:** Session 5 Statistical Approaches (11 Laboratory Programs)\n\n"
        "### Overview\n"
        "This assignment implements the complete laboratory suite for Session 5 across two core sections:\n"
        "- **Part A: Statistical Language Processing (Experiments 1-5):** Probability theory, N-gram language models, "
        "discrete-time prediction, Markov state transitions, Shannon entropy, and Probabilistic Context-Free Grammar (PCFG) parsing.\n"
        "- **Part B: Statistical Text Classification & Clustering (Experiments 6-11):** Bag-of-Words and TF-IDF categorization, "
        "Multinomial Naive Bayes, Support Vector Machines (SVM), K-Means unsupervised clustering, Centroid-based classification, "
        "and comparative algorithm performance evaluation."
    ),
    nbformat.v4.new_code_cell(
        r'''import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("All statistical NLP libraries loaded successfully.")'''
    ),
    nbformat.v4.new_markdown_cell(
        "--- \n"
        "## Part A: Statistical Language Processing\n\n"
        "### Experiment 1: Probability Theory and Language Models (Unigram, Bigram, Trigram)"
    ),
    nbformat.v4.new_code_cell(
        r'''corpus = [
    "natural language processing is interesting",
    "language models predict the next word",
    "statistical natural language processing uses probability"
]

tokens = [word for sent in corpus for word in sent.lower().split()]
unigram_counts = Counter(tokens)
total_tokens = len(tokens)

bigrams = []
trigrams = []
for sent in corpus:
    words = sent.lower().split()
    bigrams.extend(zip(words[:-1], words[1:]))
    trigrams.extend(zip(words[:-2], words[1:-1], words[2:]))

bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)

print(f"Total tokens in corpus: {total_tokens}")
print("Top 5 Unigrams with Probabilities:")
for word, count in unigram_counts.most_common(5):
    prob = count / total_tokens
    print(f"  P({word}) = {count}/{total_tokens} = {prob:.4f}")

print("\nSample Bigram Probabilities:")
for bg, count in list(bigram_counts.items())[:4]:
    prev_word_count = unigram_counts[bg[0]]
    cond_prob = count / prev_word_count
    print(f"  P({bg[1]} | {bg[0]}) = {count}/{prev_word_count} = {cond_prob:.4f}")'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiment 2: Discrete-Time Language Models (N-gram Next-Word Prediction)"
    ),
    nbformat.v4.new_code_cell(
        r'''training_text = (
    "natural language processing enables machines to understand human language. "
    "natural language processing models learn patterns from text data. "
    "deep learning models improve natural language processing performance."
)

words = training_text.lower().replace(".", "").split()
bigram_model = defaultdict(Counter)

for w1, w2 in zip(words[:-1], words[1:]):
    bigram_model[w1][w2] += 1

def predict_next_word(current_word: str):
    current_word = current_word.lower()
    if current_word not in bigram_model:
        return "Unknown (word not in vocabulary)", 0.0
    candidates = bigram_model[current_word]
    total = sum(candidates.values())
    best_word, best_count = candidates.most_common(1)[0]
    return best_word, best_count / total

test_words = ["natural", "language", "deep", "text"]
print("Discrete-Time Bigram Next-Word Predictions:")
for w in test_words:
    pred, prob = predict_next_word(w)
    print(f"  Given '{w}' -> Predicted next word: '{pred}' (Probability: {prob:.2f})")'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiment 3: Markov Models for Text Processing & State Transitions"
    ),
    nbformat.v4.new_code_cell(
        r'''# 3-State Markov Model for POS Transition (Noun -> Verb -> Adjective)
states = ["Noun", "Verb", "Adj"]
transitions = [
    ("Noun", "Verb"), ("Verb", "Noun"), ("Noun", "Verb"),
    ("Verb", "Adj"), ("Adj", "Noun"), ("Noun", "Verb")
]

trans_matrix = defaultdict(Counter)
for s1, s2 in transitions:
    trans_matrix[s1][s2] += 1

df_trans = pd.DataFrame(index=states, columns=states).fillna(0.0)
for s1 in states:
    total = sum(trans_matrix[s1].values())
    if total > 0:
        for s2 in states:
            df_trans.loc[s1, s2] = trans_matrix[s1][s2] / total

print("Markov State Transition Probability Matrix:")
display(df_trans)'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiment 4: Entropy and Information Theory in NLP (Shannon Entropy & Cross-Entropy)"
    ),
    nbformat.v4.new_code_cell(
        r'''def calculate_entropy(text_tokens):
    counts = Counter(text_tokens)
    total = len(text_tokens)
    entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
    return entropy

corpus_A = "the cat sat on the mat".split()
corpus_B = "alpha beta gamma delta epsilon zeta eta theta".split()

entropy_A = calculate_entropy(corpus_A)
entropy_B = calculate_entropy(corpus_B)

print(f"Corpus A (repetitive tokens) Entropy: {entropy_A:.4f} bits/word (Perplexity: {2**entropy_A:.2f})")
print(f"Corpus B (uniform tokens) Entropy:    {entropy_B:.4f} bits/word (Perplexity: {2**entropy_B:.2f})")'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiment 5: Statistical Parsing using Probabilistic Context-Free Grammar (PCFG)"
    ),
    nbformat.v4.new_code_cell(
        r'''pcfg_grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.7] | 'I' [0.3]
    VP -> V NP [0.8] | V [0.2]
    Det -> 'the' [0.6] | 'a' [0.4]
    N -> 'cat' [0.5] | 'dog' [0.5]
    V -> 'chased' [0.7] | 'saw' [0.3]
""")

viterbi = ViterbiParser(pcfg_grammar)
parse_sentence = "the cat chased a dog".split()

print("Viterbi Most Probable Parse Tree:")
parses = list(viterbi.parse(parse_sentence))
if parses:
    for tree in parses:
        print(f"Log probability: {tree.prob():.6e}")
        print(tree)
else:
    print("No valid parse found under current grammar.")'''
    ),
    nbformat.v4.new_markdown_cell(
        "--- \n"
        "## Part B: Statistical Text Classification & Clustering\n\n"
        "### Experiment 6: Text Categorization Using Machine Learning (BoW & TF-IDF)"
    ),
    nbformat.v4.new_code_cell(
        r'''text_dataset = [
    ("Python code and machine learning models are powerful.", "Tech"),
    ("Software developers use git and docker containers.", "Tech"),
    ("Neural network architectures optimize gradient descent.", "Tech"),
    ("The football match ended in a thrilling draw.", "Sports"),
    ("The cricket tournament featured outstanding bowling.", "Sports"),
    ("Athletes train hard for the upcoming Olympic games.", "Sports"),
    ("Doctors recommend healthy nutrition and regular exercise.", "Health"),
    ("Medical research discovered new clinical vaccines.", "Health"),
    ("Hospitals improved emergency patient care standards.", "Health")
]

df_cat = pd.DataFrame(text_dataset, columns=["Text", "Category"])
tfidf_vec = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf_vec.fit_transform(df_cat["Text"])

print(f"Document-Term Matrix Shape: {X_tfidf.shape} (9 documents x {X_tfidf.shape[1]} features)")
print("Sample extracted TF-IDF feature vocabulary:", list(tfidf_vec.get_feature_names_out())[:10])'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiments 7, 8, 10 & 11: Text Classification Algorithms & Comparative Analysis"
    ),
    nbformat.v4.new_code_cell(
        r'''# Benchmark dataset for comparative classification
X = df_cat["Text"]
y = df_cat["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

vec = TfidfVectorizer(stop_words="english")
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

models = {
    "Naive Bayes (MultinomialNB)": MultinomialNB(),
    "Support Vector Machine (LinearSVC)": LinearSVC(random_state=42),
    "Centroid Classifier (NearestCentroid)": NearestCentroid()
}

results = []
for name, clf in models.items():
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    results.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

df_results = pd.DataFrame(results)
print("Comparative Performance Analysis (Session 5 - Exp 11):")
display(df_results)'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Experiment 9: Text Clustering Using K-Means (Unsupervised Learning)"
    ),
    nbformat.v4.new_code_cell(
        r'''kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_tfidf)

df_cat["Cluster"] = cluster_labels
print("K-Means Cluster Assignments vs True Category:")
display(df_cat[["Text", "Category", "Cluster"]])

terms = tfidf_vec.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop 3 Keywords per Cluster:")
for i in range(3):
    top_words = [terms[ind] for ind in order_centroids[i, :3]]
    print(f"  Cluster {i}: {', '.join(top_words)}")'''
    ),
    nbformat.v4.new_markdown_cell(
        "### Summary of Day 4 Laboratory Findings\n"
        "1. **Language Modeling:** Unigram and bigram models capture word frequencies and short-range dependencies with low compute.\n"
        "2. **Markov Chains:** Useful for state-transition modeling (e.g. POS tagging).\n"
        "3. **Statistical Parsing (PCFG):** Disambiguates syntactic parse trees by choosing the highest joint probability tree.\n"
        "4. **Text Classification:** Naive Bayes, Linear SVM, and Nearest Centroid all provide strong baselines on TF-IDF vectors."
    )
]

nb.cells = cells

client = NotebookClient(nb, timeout=60, kernel_name="python3")
print("Executing Assignment 4 notebook cells...")
client.execute()

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook successfully written and executed: {NOTEBOOK_PATH}")
