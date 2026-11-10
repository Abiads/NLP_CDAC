"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 5: Comparison of Constituency and Dependency Parsing
File:   04. Constituency vs Dependency Parsing Comparison.py
=============================================================================

CONSTITUENCY VS DEPENDENCY PARSING:
-----------------------------------
1. Constituency Parsing (Phrase Structure Grammar):
   - Focuses on nested phrase components (NP, VP, PP).
   - Generates non-terminal nodes in addition to leaf words.
   - Tied to fixed word-order languages (e.g., English SVO).
   - Ambiguities arise from alternative phrase bracketings.

2. Dependency Parsing (Relational Grammar):
   - Focuses on direct relationships between words (Governor -> Dependent).
   - No non-terminal nodes; only words as nodes in a directed tree.
   - Very effective for morphologically rich and free word-order languages.
   - Directly reveals semantic roles (e.g. subject, direct object).

PP-ATTACHMENT AMBIGUITY DEMONSTRATION:
--------------------------------------
"She ate the cake with a fork"
  - Reading 1: ate [with a fork]       -> Instrument used to eat (high plausibility)
  - Reading 2: cake [with a fork]      -> Cake has a fork on it (low plausibility)
=============================================================================
"""

import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser
import spacy

# Step 1: Define CFG for Constituency Parsing
grammar_str = """
    S -> NP VP
    NP -> Det N | Pro | Det N PP
    VP -> V NP | V NP PP
    PP -> P NP
    
    Det -> 'the' | 'a'
    Pro -> 'she'
    N -> 'cake' | 'fork' | 'student' | 'book' | 'teacher'
    V -> 'ate' | 'reads' | 'explains'
    P -> 'with'
"""

const_grammar = CFG.fromstring(grammar_str)
const_parser = RecursiveDescentParser(const_grammar)

# Step 2: Load spaCy Dependency Parser
nlp = spacy.load("en_core_web_sm")

# Step 3: Test Sentence
test_sentence = "the student reads a book"
ambiguous_sentence = "she ate the cake with a fork"

print("=" * 75)
print("EXPERIMENT 5: CONSTITUENCY VS DEPENDENCY PARSING COMPARISON")
print("=" * 75)

for sent in [test_sentence, ambiguous_sentence]:
    print("\n" + "=" * 75)
    print(f"Target Sentence: \"{sent}\"")
    print("=" * 75)
    
    # -----------------------------------------------------------------------
    # Part A: Constituency Parsing Output
    # -----------------------------------------------------------------------
    print("\n[A] CONSTITUENCY PARSING (Phrase Structure Decomposition)")
    print("-" * 60)
    tokens = sent.lower().split()
    try:
        trees = list(const_parser.parse(tokens))
        if trees:
            print(f"Found {len(trees)} constituency parse tree(s):")
            for idx, t in enumerate(trees, 1):
                print(f"\nParse Tree #{idx}:")
                t.pretty_print()
        else:
            print("No constituency parse generated for current grammar.")
    except Exception as e:
        print(f"Constituency parse error: {e}")
        
    # -----------------------------------------------------------------------
    # Part B: Dependency Parsing Output
    # -----------------------------------------------------------------------
    print("\n[B] DEPENDENCY PARSING (Head-Dependent Directed Relations)")
    print("-" * 60)
    doc = nlp(sent)
    print(f"{'Word':<12} | {'POS':<6} | {'Dependency':<12} | {'Head Word':<12} | {'Head POS'}")
    print("-" * 60)
    for tok in doc:
        print(f"{tok.text:<12} | {tok.pos_:<6} | {tok.dep_:<12} | {tok.head.text:<12} | {tok.head.pos_}")

# ---------------------------------------------------------------------------
# Part C: Key Architectural Differences Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("COMPARATIVE SUMMARY TABLE")
print("=" * 75)
summary_data = [
    ("Core Principle", "Phrase structure & nested bracketings", "Head-to-dependent directed relations"),
    ("Nodes in Tree", "Non-terminals (NP, VP) + leaf words", "Only words from the sentence"),
    ("Complexity", "O(n^3) (CYK / Chart parsing)", "Linear O(n) (Shift-reduce / Transition)"),
    ("Free Word Order", "Struggles with non-projective word order", "Naturally models free word order"),
    ("Downstream Use", "Formal language parsing, compilers", "Information Extraction, Relation Extraction")
]

print(f"{'Dimension':<20} | {'Constituency Parsing':<38} | {'Dependency Parsing'}")
print("-" * 85)
for dim, c_val, d_val in summary_data:
    print(f"{dim:<20} | {c_val:<38} | {d_val}")

print("=" * 75)
print("[Summary] Constituency models phrase groupings; Dependency models direct syntactic relations.")
print("=" * 75)

