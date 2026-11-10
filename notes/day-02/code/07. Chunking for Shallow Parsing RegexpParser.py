"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Chunking for Shallow Parsing using NLTK RegexpParser
File:   07_chunking_shallow_parsing_regexpparser.py
=============================================================================

WHAT IS CHUNKING (SHALLOW PARSING)?
-----------------------------------
Chunking (also called Shallow Parsing) is a technique in NLP that groups
consecutive tokens into flat, non-overlapping grammatical phrases such as:
  - Noun Phrases (NP): e.g., "the magical park", "a long waiting line"
  - Verb Phrases (VP): e.g., "was waiting", "has visited"
  - Prepositional Phrases (PP): e.g., "in the morning", "at Disneyland"

Unlike Deep Parsing (which creates full hierarchical trees of full sentences),
Shallow Parsing identifies major phrases without analyzing complex nested clauses.

THE NOUN PHRASE (NP) CHUNK GRAMMAR:
-----------------------------------
A standard Noun Phrase chunk pattern:
  NP: {<DT>?<JJ>*<NN.*>+}

Meaning:
  <DT>?   : Optional Determiner (the, a, an, this)
  <JJ>*   : Zero or more Adjectives (magical, long, exciting)
  <NN.*>+ : One or more Nouns (park, queue, Space Mountain)
=============================================================================
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# Download required tokenizer and tagger models
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ---------------------------------------------------------------------------
# Step 1: Define the Sample Sentence
# ---------------------------------------------------------------------------
sentence = "The magical park has a long waiting line for the exciting roller coaster."
print("=" * 70)
print("1. ORIGINAL SENTENCE & POS TAGGING")
print("=" * 70)
print(f"Sentence: \"{sentence}\"\n")

# Tokenize and POS tag the sentence
tokens = word_tokenize(sentence)
tagged_tokens = pos_tag(tokens)

print("POS Tagged Tokens:")
for word, tag in tagged_tokens:
    print(f"  {word:<15} --> {tag}")

# ---------------------------------------------------------------------------
# Step 2: Define Chunk Grammar for Determiner + Adjective + Noun
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. NOUN PHRASE (NP) CHUNKING WITH REGEXP PARSER")
print("=" * 70)

# Grammar rule requested: Determiner (<DT>), Adjectives (<JJ>), and Nouns (<NN.*>)
# Chunk rule: optional DT + zero or more JJ + one or more NN
np_grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
"""

# Create the RegexpParser with our custom grammar
np_parser = RegexpParser(np_grammar)

# Parse the POS-tagged tokens into a Chunk Tree
chunk_tree = np_parser.parse(tagged_tokens)

print("Chunk Tree Output (Text representation):")
print(chunk_tree)

# ---------------------------------------------------------------------------
# Step 3: Extract and Display Noun Phrase (NP) Chunks Cleanly
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. EXTRACTED NOUN PHRASES (NP)")
print("=" * 70)

extracted_chunks = []
for subtree in chunk_tree.subtrees():
    if subtree.label() == "NP":
        # Extract the words belonging to this NP chunk
        chunk_words = [word for word, tag in subtree.leaves()]
        chunk_text = " ".join(chunk_words)
        extracted_chunks.append(chunk_text)
        print(f"  Found NP: \"{chunk_text}\"")

# ---------------------------------------------------------------------------
# Step 4: Multi-Phrase Chunking (NP, VP, and PP)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. COMPREHENSIVE SHALLOW PARSER (NP, VP, PP)")
print("=" * 70)

# NP: optional det + zero or more adj + one or more nouns
# VP: any sequence of verbs
# PP: preposition
multi_grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    VP: {<VB.*>+}
    PP: {<IN>}
"""

multi_parser = RegexpParser(multi_grammar)
multi_tree = multi_parser.parse(tagged_tokens)

print(f"{'Phrase Type':<15} | {'Tokens in Phrase'}")
print("-" * 50)
for subtree in multi_tree.subtrees():
    if subtree.label() in ["NP", "VP", "PP"]:
        phrase_words = " ".join(word for word, tag in subtree.leaves())
        print(f"{subtree.label():<15} | \"{phrase_words}\"")

# ---------------------------------------------------------------------------
# Step 5: Chinking (Excluding Patterns from a Chunk)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. CHINKING DEMO (Carving Out Patterns)")
print("=" * 70)
print("Chinking means defining what to REMOVE from a chunk.")
print("Below we chunk everything ({<.*>+}) and chink out verbs/prepositions (}<VB.*|IN>+{):\n")

# Step 1: Chunk the entire sentence ({<.*>+})
# Step 2: Chink (exclude) verbs and prepositions (}<VB.*|IN>+{)
chink_grammar = r"""
    CHUNK: {<.*>+}
           }<VB.*|IN>+{
"""

chink_parser = RegexpParser(chink_grammar)
chink_tree = chink_parser.parse(tagged_tokens)

for subtree in chink_tree.subtrees():
    if subtree.label() == "CHUNK":
        words_in_chunk = " ".join(word for word, tag in subtree.leaves())
        print(f"  Remaining Chunk: \"{words_in_chunk}\"")

print("=" * 70)
print("[Summary] Chunking segments flat syntactic units like NP: <DT>?<JJ>*<NN>.")
print("=" * 70)
