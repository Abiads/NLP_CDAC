"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Stemming using NLTK PorterStemmer
File:   4_stemming_using_porter_stemmer.py
=============================================================================

WHAT IS STEMMING?
-----------------
Stemming is a rule-based process in NLP that chops off common prefixes and
suffixes (like '-ing', '-ed', '-ly', '-s') from words to reduce them to their
root base form called a "stem".

Example:
  'connect', 'connecting', 'connected', 'connection' -> all become 'connect'

WHY DO WE USE STEMMING?
-----------------------
1. Reduces vocabulary size (shinks document matrix).
2. Groups different forms of the same word together so search engines and
   classifiers treat them as the same feature.

THE PORTER STEMMER ALGORITHM:
-----------------------------
Developed by Martin Porter in 1980, it applies a pipeline of 5 sequential
rule-based phases to strip English suffixes.
=============================================================================
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required tokenizer model (punkt)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Step 1: Initialize the Porter Stemmer
# ---------------------------------------------------------------------------
porter = PorterStemmer()

# ---------------------------------------------------------------------------
# Step 2: Stemming Individual Words
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. STEMMING INDIVIDUAL WORDS (Suffix Stripping)")
print("=" * 65)

sample_words = [
    "connection", "connecting", "connections", "connected",
    "program", "programming", "programmer", "programs",
    "easily", "fairly", "relativity", "studies"
]

print(f"{'Original Word':<20} --> {'Porter Stem':<15}")
print("-" * 40)
for word in sample_words:
    stemmed_word = porter.stem(word)
    print(f"{word:<20} --> {stemmed_word:<15}")

# ---------------------------------------------------------------------------
# Step 3: Stemming an Entire Sentence
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. STEMMING A COMPLETE SENTENCE")
print("=" * 65)

sentence = "The students are studying and writing programs for natural language processing."
print("Original Sentence:")
print(sentence)

# Step 3a: Split sentence into individual words (tokens)
tokens = word_tokenize(sentence)

# Step 3b: Apply stemmer to each word in the token list
stemmed_tokens = [porter.stem(token) for token in tokens]

# Step 3c: Reconstruct the sentence with stems
stemmed_sentence = " ".join(stemmed_tokens)

print("\nTokens:")
print(tokens)

print("\nStemmed Sentence:")
print(stemmed_sentence)

# ---------------------------------------------------------------------------
# Step 4: Understanding Limitations (Non-Word Stems)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. PORTER STEMMER LIMITATIONS (Under-stemming & Over-stemming)")
print("=" * 65)

tricky_words = ["happy", "university", "troubled", "organize"]

print("Notice that Porter Stemmer chops endings by rules, so the output")
print("is not always a valid English dictionary word:\n")
for word in tricky_words:
    print(f"  '{word}' -> stem: '{porter.stem(word)}' (Notice spelling)")

print("\n[Summary] Stemming is fast and simple, but for grammatically accurate")
print("dictionary roots, Lemmatization (using WordNet) is used instead.")
print("=" * 65)
