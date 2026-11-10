"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Lemmatization using NLTK WordNetLemmatizer
File:   05_lemmatization_using_wordnet.py
=============================================================================

WHAT IS LEMMATIZATION?
----------------------
Lemmatization is a linguistically sound process that converts inflected forms
of a word into its meaningful base or dictionary root form, called a "lemma".

HOW IS LEMMATIZATION DIFFERENT FROM STEMMING?
---------------------------------------------
1. Stemming: Uses crude rules to chop off prefixes/suffixes. Often results
   in non-dictionary words (e.g. 'relativity' -> 'relativ').
2. Lemmatization: Uses vocabulary and morphological analysis (WordNet). Always
   returns a legitimate English dictionary word (e.g. 'better' -> 'good',
   'children' -> 'child').

THE IMPORTANCE OF PART-OF-SPEECH (POS) IN LEMMATIZATION:
--------------------------------------------------------
By default, NLTK's WordNetLemmatizer treats every word as a NOUN.
To lemmatize verbs ('running' -> 'run') or adjectives ('better' -> 'good'),
we must pass the correct POS tag to the lemmatizer!
=============================================================================
"""

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download required NLTK corpora and models
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Initialize the lemmatizer and stemmer for comparison
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# ---------------------------------------------------------------------------
# Step 1: Basic Lemmatization on Nouns (Default behavior)
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. BASIC NOUN LEMMATIZATION (Default POS = Noun)")
print("=" * 65)

noun_words = ["children", "feet", "wolves", "cars", "queues", "mice"]

print(f"{'Original Word':<15} --> {'Porter Stem':<15} --> {'WordNet Lemma':<15}")
print("-" * 50)
for word in noun_words:
    stem_result = stemmer.stem(word)
    lemma_result = lemmatizer.lemmatize(word)  # default is pos='n'
    print(f"{word:<15} --> {stem_result:<15} --> {lemma_result:<15}")

# ---------------------------------------------------------------------------
# Step 2: Why Part-of-Speech (POS) Matters for Verbs and Adjectives
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. EFFECT OF PART-OF-SPEECH (POS) ON LEMMATIZATION")
print("=" * 65)

# Example: WordNet default treats 'was' as a noun!
print("Default lemmatize('was')           :", lemmatizer.lemmatize("was"))
print("Correct lemmatize('was', pos='v')   :", lemmatizer.lemmatize("was", pos=wordnet.VERB))

print("\nDefault lemmatize('running')       :", lemmatizer.lemmatize("running"))
print("Correct lemmatize('running', pos='v'):", lemmatizer.lemmatize("running", pos=wordnet.VERB))

print("\nDefault lemmatize('better')        :", lemmatizer.lemmatize("better"))
print("Correct lemmatize('better', pos='a') :", lemmatizer.lemmatize("better", pos=wordnet.ADJ))

# ---------------------------------------------------------------------------
# Step 3: Automated POS-Aware Lemmatization Function
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. AUTOMATED SENTENCE LEMMATIZATION (With POS Tagging)")
print("=" * 65)

def get_wordnet_pos(penn_treebank_tag):
    """
    Maps Penn Treebank POS tags (from nltk.pos_tag) to WordNet POS constants:
      - 'J...' -> ADJ
      - 'V...' -> VERB
      - 'N...' -> NOUN
      - 'R...' -> ADV
      - default -> NOUN
    """
    if penn_treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_treebank_tag.startswith('V'):
        return wordnet.VERB
    elif penn_treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    """Tokenizes, POS-tags, and lemmatizes every word with its proper grammatical category."""
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    
    lemmatized_output = []
    for word, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        lemmatized_output.append(lemma)
        
    return lemmatized_output

# Test sentence from Ma'am's Session 3 Slide 26
sample_sentence = "The children were waiting in long queues at the park."
print("Original Sentence:")
print(sample_sentence)

tokens = word_tokenize(sample_sentence)
stemmed_sentence = [stemmer.stem(w) for w in tokens]
lemmatized_sentence = lemmatize_text(sample_sentence)

print("\nTokens            :", tokens)
print("Porter Stemmed    :", stemmed_sentence)
print("WordNet Lemmatized:", lemmatized_sentence)

# ---------------------------------------------------------------------------
# Step 4: Side-by-Side Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. COMPARISON SUMMARY: TOKEN vs STEM vs LEMMA")
print("=" * 65)

test_words = [
    ("children", "N"),
    ("were", "V"),
    ("waiting", "V"),
    ("queues", "N"),
    ("better", "J"),
    ("fairly", "R"),
    ("troubled", "V")
]

print(f"{'Word':<12} {'POS Tag':<10} {'Porter Stem':<15} {'WordNet Lemma':<15}")
print("-" * 55)
for word, pos_letter in test_words:
    wn_p = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}[pos_letter]
    s_val = stemmer.stem(word)
    l_val = lemmatizer.lemmatize(word, pos=wn_p)
    print(f"{word:<12} {pos_letter:<10} {s_val:<15} {l_val:<15}")

print("=" * 65)
