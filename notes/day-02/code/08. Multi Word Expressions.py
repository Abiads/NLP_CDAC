"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Multi-Word Expressions (MWEs) using NLTK MWETokenizer
File:   08_multi_word_expressions.py
=============================================================================

WHAT IS A MULTI-WORD EXPRESSION (MWE)?
--------------------------------------
A Multi-Word Expression (MWE) is a sequence of words that acts as a single
linguistic, syntactic, or semantic unit.

If we tokenize words individually, we lose their combined meaning:
  - "Disneyland Paris" -> "Disneyland", "Paris" (Loses entity unity)
  - "roller coaster"   -> "roller", "coaster"   (Loses compound noun unity)
  - "had a blast"      -> "had", "a", "blast"   (Loses idiomatic meaning)

TYPES OF MWEs (Covered in Session 3 Slide 15):
----------------------------------------------
1. Compound Nouns / Proper Names : "Disneyland Paris", "Space Mountain", "ice cream"
2. Phrasal Verbs                 : "end up", "look forward to", "check in"
3. Idiomatic Expressions        : "had a blast", "by and large", "spill the beans"

HOW DOES NLTK HANDLE MWEs?
--------------------------
NLTK provides `MWETokenizer` which takes pre-tokenized text and merges predefined
multi-word tuples into single connected tokens using an underscore separator:
  ('Disneyland', 'Paris') -> 'Disneyland_Paris'
  ('roller', 'coaster')   -> 'roller_coaster'
=============================================================================
"""

import nltk
from nltk.tokenize import word_tokenize, MWETokenizer

# Download required punkt tokenizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Step 1: Define Sample Review Sentence with Multi-Word Expressions
# ---------------------------------------------------------------------------
review_text = (
    "When you visit Disneyland Paris, you will end up comparing it to other parks. "
    "We rode the Space Mountain roller coaster and absolutely had a blast!"
)

print("=" * 70)
print("1. ORIGINAL REVIEW TEXT")
print("=" * 70)
print(f"\"{review_text}\"")

# ---------------------------------------------------------------------------
# Step 2: Standard Word Tokenization (Without MWE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. STANDARD WORD TOKENIZATION (Without MWE)")
print("=" * 70)

standard_tokens = word_tokenize(review_text)
print("Notice how compound entities and idioms are chopped into isolated pieces:")
print(standard_tokens)

# ---------------------------------------------------------------------------
# Step 3: Define Multi-Word Expression (MWE) Tuples
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. DEFINING MWE VOCABULARY & BUILDING MWETokenizer")
print("=" * 70)

# Define list of multi-word expressions as tuples of individual tokens
mwe_tuples = [
    ("Disneyland", "Paris"),
    ("Space", "Mountain"),
    ("roller", "coaster"),
    ("end", "up"),
    ("had", "a", "blast")
]

print("Registered Multi-Word Expressions:")
for mwe in mwe_tuples:
    print(f"  Tuple: {mwe} --> Target: '{'_'.join(mwe)}'")

# Initialize the MWETokenizer with our custom tuples
mwe_tokenizer = MWETokenizer(mwe_tuples, separator='_')

# We can also add new multi-word expressions dynamically using add_mwe()
mwe_tokenizer.add_mwe(("Hong", "Kong"))
mwe_tokenizer.add_mwe(("fast", "pass"))

# ---------------------------------------------------------------------------
# Step 4: Tokenizing with MWE Preservation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MWE-PRESERVED TOKENIZATION OUTPUT")
print("=" * 70)

# Pass the standard tokens through the MWE tokenizer
mwe_tokens = mwe_tokenizer.tokenize(standard_tokens)

print("Tokens after MWE processing (Multi-words fused into single units):")
print(mwe_tokens)

# ---------------------------------------------------------------------------
# Step 5: Side-by-Side Comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. BEFORE vs AFTER MWE TOKENIZATION COMPARISON")
print("=" * 70)

print(f"{'Multi-Word Expression':<25} | {'Standard Tokenizer':<25} | {'MWE Tokenizer'}")
print("-" * 75)

comparisons = [
    ("Proper Name", "Disneyland, Paris", "Disneyland_Paris"),
    ("Compound Noun", "roller, coaster", "roller_coaster"),
    ("Attraction Name", "Space, Mountain", "Space_Mountain"),
    ("Phrasal Verb", "end, up", "end_up"),
    ("Idiom", "had, a, blast", "had_a_blast")
]

for label, before, after in comparisons:
    print(f"{label:<25} | {before:<25} | {after}")

print("=" * 70)
print("[Summary] MWETokenizer prevents entity fragmentation during NLP preprocessing.")
print("=" * 70)
