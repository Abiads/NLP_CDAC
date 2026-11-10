"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Experiment 11: Word Sense Disambiguation (WSD) using Lesk Algorithm
File:   11. Word Sense Disambiguation using Lesk.py
=============================================================================

WHAT IS WORD SENSE DISAMBIGUATION (WSD)?
----------------------------------------
Word Sense Disambiguation (WSD) is an open problem in Natural Language
Processing that aims to identify which sense (meaning) of a polysemous word
is used in a particular sentence or context.

For example, the word "bank":
  - "The fisherman sat on the bank of the river."  -> Sloping land beside water
  - "She deposited money in the bank."             -> Financial institution

HOW DOES THE LESK ALGORITHM WORK?
---------------------------------
Proposed by Michael E. Lesk (1986), the Simplified Lesk Algorithm selects
the synset (sense) whose dictionary definition (gloss) and examples share the
maximum word overlap (intersection) with the context words surrounding the
target ambiguous word:

  Score(Sense_i) = | Context_Words INTERSECT (Gloss_Words_i UNION Example_Words_i) |

The synset with the highest overlap score is selected as the predicted sense.

DATASET & EXAMPLES IN THIS SCRIPT:
----------------------------------
1. Classic ambiguous word demonstration ("bank" in financial vs river contexts).
2. Disneyland review demonstration ("park" as amusement venue vs vehicle action).
3. Detailed Synset Definition and Example inspection from WordNet.
=============================================================================
"""

import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# Download required NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("=" * 75)
print("EXPERIMENT 11: WORD SENSE DISAMBIGUATION (WSD) - LESK ALGORITHM")
print("=" * 75)

# ===========================================================================
# Part 1: Classic Ambiguous Word ("bank")
# ===========================================================================
print("\n" + "=" * 75)
print("1. CLASSIC HOMONYM AMBIGUITY: 'bank'")
print("=" * 75)

sentence1 = "The fisherman sat on the bank of the river catching fish."
sentence2 = "She went to the bank to deposit her monthly salary and open an account."
target_word1 = "bank"

tokens1 = word_tokenize(sentence1)
tokens2 = word_tokenize(sentence2)

# Apply Lesk algorithm
sense1 = lesk(tokens1, target_word1, pos='n')
sense2 = lesk(tokens2, target_word1, pos='n')

print(f"Sentence 1 : \"{sentence1}\"")
print(f"Target Word: '{target_word1}'")
if sense1:
    print(f"Predicted Synset : {sense1.name()}")
    print(f"Definition       : {sense1.definition()}")
else:
    print("No sense identified.")

print("-" * 75)
print(f"Sentence 2 : \"{sentence2}\"")
print(f"Target Word: '{target_word1}'")
if sense2:
    print(f"Predicted Synset : {sense2.name()}")
    print(f"Definition       : {sense2.definition()}")
else:
    print("No sense identified.")

# ===========================================================================
# Part 2: Disneyland Domain Ambiguity: "park" (Noun vs Verb)
# ===========================================================================
print("\n" + "=" * 75)
print("2. DOMAIN SENSE DISAMBIGUATION: 'park'")
print("=" * 75)

disney_sent1 = "We visited the theme park near Disneyland to ride roller coasters."
disney_sent2 = "The driver had to park the car in the underground parking lot."
target_word2 = "park"

tokens_disney1 = word_tokenize(disney_sent1)
tokens_disney2 = word_tokenize(disney_sent2)

# Ambiguous between Noun (amusement area) and Verb (maneuver vehicle)
sense_disney1 = lesk(tokens_disney1, target_word2, pos='n')
sense_disney2 = lesk(tokens_disney2, target_word2, pos='v')

print(f"Sentence 1 (Noun) : \"{disney_sent1}\"")
print(f"Predicted Sense   : {sense_disney1.name() if sense_disney1 else 'None'}")
if sense_disney1:
    print(f"Definition        : {sense_disney1.definition()}")

print("-" * 75)
print(f"Sentence 2 (Verb) : \"{disney_sent2}\"")
print(f"Predicted Sense   : {sense_disney2.name() if sense_disney2 else 'None'}")
if sense_disney2:
    print(f"Definition        : {sense_disney2.definition()}")

# ===========================================================================
# Part 3: Inspecting Available Synsets in WordNet
# ===========================================================================
print("\n" + "=" * 75)
print("3. ALL AVAILABLE SENSES IN WORDNET FOR 'park'")
print("=" * 75)

synsets_park = wn.synsets("park")
print(f"Total synsets found for 'park': {len(synsets_park)}\n")

print(f"{'Synset':<20} | {'POS':<6} | {'Definition'}")
print("-" * 75)
for s in synsets_park[:6]:  # Display top 6 senses
    pos_type = "Noun" if s.pos() == 'n' else ("Verb" if s.pos() == 'v' else s.pos())
    print(f"{s.name():<20} | {pos_type:<6} | {s.definition()}")

# ===========================================================================
# Part 4: Real Dataset Application on Disneyland Reviews
# ===========================================================================
print("\n" + "=" * 75)
print("4. REAL DATASET APPLICATION (Disneyland Reviews Dataset)")
print("=" * 75)

csv_candidates = [
    "DisneylandReviews.csv",
    os.path.join("notes", "day-02", "DisneylandReviews.csv"),
    os.path.join("..", "DisneylandReviews.csv"),
    os.path.join("..", "..", "notes", "day-02", "DisneylandReviews.csv")
]

csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

if csv_path:
    print(f"Loading reviews from: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin-1', nrows=3)
    
    review_sample = str(df['Review_Text'][0])
    print(f"\nSample Review Text:\n\"{review_sample[:250]}...\"\n")
    
    # Disambiguate key words in review
    target_terms = ["park", "line", "ride"]
    review_tokens = word_tokenize(review_sample)
    
    print(f"{'Target Word':<15} | {'Predicted Synset':<22} | {'Context Definition'}")
    print("-" * 75)
    for term in target_terms:
        term_sense = lesk(review_tokens, term)
        if term_sense:
            defn = term_sense.definition()
            print(f"{term:<15} | {term_sense.name():<22} | {defn[:45]}...")
        else:
            print(f"{term:<15} | {'None':<22} | No sense detected.")
else:
    print("[Note] DisneylandReviews.csv not found in local path.")

print("\n" + "=" * 75)
print("[Summary] Lesk algorithm effectively resolves polysemy by maximizing context-gloss overlap.")
print("=" * 75)
