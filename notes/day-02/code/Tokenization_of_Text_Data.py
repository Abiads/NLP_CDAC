"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Text Preprocessing & Tokenization using NLTK and Pandas
File:   Tokenization_of_Text_Data.py
=============================================================================

WHAT IS TOKENIZATION?
---------------------
Tokenization is the very first step in Natural Language Processing (NLP).
It means breaking down a large block of unstructured text into smaller,
meaningful pieces called "tokens":
  1. Sentence Tokenization: Breaking a paragraph into individual sentences.
  2. Word Tokenization:     Breaking a sentence into individual words & punctuation.

WHY NOT JUST USE python's text.split(' ') OR text.split('.')?
------------------------------------------------------------
- Simple .split('.') fails on abbreviations like "Dr. Sharma", "e.g.", "U.S.A.",
  or numbers like "3.14".
- Simple .split(' ') fails on punctuation: "hello," becomes "hello," (with comma attached)
  instead of ["hello", ","].
- NLTK's `punkt` tokenizer is trained to understand punctuation and abbreviations!
=============================================================================
"""

import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# Step 1: Download NLTK Tokenizer Models (Punkt)
# ---------------------------------------------------------------------------
# 'punkt' contains the pre-trained unsupervised models for sentence boundary detection
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Step 2: Load the Dataset (Disneyland Customer Reviews)
# ---------------------------------------------------------------------------
# In NLP, real-world customer reviews often contain special characters, accents,
# or emojis. We use encoding='latin1' to prevent UnicodeDecodeError.
csv_filename = 'DisneylandReviews.csv'

# Check if the CSV exists in current folder, else use a fallback review
if os.path.exists(csv_filename):
    print(f"[DIR] Loading reviews from '{csv_filename}'...")
    df = pd.read_csv(csv_filename, encoding='latin1')
    # Pick the first customer review text (Row 0, column 'Review_Text')
    review = str(df['Review_Text'][0])
else:
    print(f"[WARNING]  '{csv_filename}' not found. Using sample Disneyland review...")
    review = (
        "If you've ever been to Disneyland Paris you'll end up comparing the two. "
        "I visited Disney Hong Kong on an extremely hot day in May, but still had a blast! "
        "The park is relatively small compared to others, but it makes up for it with shorter "
        "wait times and great attractions like Mystic Manor and Big Grizzly Mountain. "
        "Don't miss the fireworks at night!"
    )

# ---------------------------------------------------------------------------
# Step 3: Display the Raw Original Review
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. ORIGINAL REVIEW (Raw Text)")
print("=" * 70)
print(review)

# ---------------------------------------------------------------------------
# Step 4: Sentence Tokenization (Paragraph -> Sentences)
# ---------------------------------------------------------------------------
# sent_tokenize uses punctuation, capitalization, and abbreviation rules
# to split a paragraph into clean, complete sentences.
print("\n" + "=" * 70)
print("2. SENTENCE TOKENIZATION (sent_tokenize)")
print("=" * 70)
sentences = sent_tokenize(review)

for index, sentence in enumerate(sentences, start=1):
    print(f"  Sentence {index}: \"{sentence}\"")

# ---------------------------------------------------------------------------
# Step 5: Word Tokenization (Sentence -> Words & Punctuation)
# ---------------------------------------------------------------------------
# word_tokenize separates words and separates punctuation marks as distinct tokens.
# Notice how it intelligently splits contractions:
#   "you've" -> ['you', "'ve"]
#   "you'll" -> ['you', "'ll"]
#   "don't"  -> ['do', "n't"]
print("\n" + "=" * 70)
print("3. WORD TOKENIZATION (word_tokenize)")
print("=" * 70)
words = word_tokenize(review)
print("List of Tokens:")
print(words)

# ---------------------------------------------------------------------------
# Step 6: Corpus Statistics (Basic Text Analytics)
# ---------------------------------------------------------------------------
# Calculating basic counts helps understand document length and vocabulary
print("\n" + "=" * 70)
print("4. SUMMARY STATISTICS")
print("=" * 70)
print("Total Sentences :", len(sentences))
print("Total Words     :", len(words))
avg_words_per_sentence = len(words) / len(sentences) if len(sentences) > 0 else 0
print(f"Avg Words/Sent  : {avg_words_per_sentence:.1f}")
print("=" * 70)
