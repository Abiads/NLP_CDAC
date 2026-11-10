"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 3: Dependency Parsing using spaCy
File:   03. Dependency Parsing with spaCy.py
=============================================================================

WHAT IS DEPENDENCY PARSING?
---------------------------
Unlike Constituency Parsing (which breaks sentences into nested non-terminal
phrases like NP and VP), Dependency Parsing links words directly to each other
via typed directed binary grammatical relations:

  Head (Governor) ----[ Dependency Relation ]----> Dependent (Modifier)

KEY DEPENDENCY LABELS IN SPACY / UNIVERSAL DEPENDENCIES:
--------------------------------------------------------
  - nsubj  : Nominal Subject (e.g., 'student' -> 'reads')
  - ROOT   : Central verb / predicate of the sentence
  - dobj   : Direct Object (e.g., 'book' -> 'reads')
  - det    : Determiner (e.g., 'The' -> 'student')
  - amod   : Adjectival Modifier (e.g., 'excited' -> 'visitors')
  - prep   : Prepositional Modifier (e.g., 'in' -> 'reads')
  - pobj   : Object of Preposition (e.g., 'library' -> 'in')

WHY DEPENDENCY PARSING IS WIDELY USED:
--------------------------------------
1. Directly captures semantic relations (who did what to whom).
2. Works naturally with free-word-order languages (such as Indian languages
   using Paninian karaka relations).
3. Highly efficient linear-time transition-based parsing.
=============================================================================
"""

import spacy
from spacy import displacy

# Step 1: Load English language pipeline
nlp = spacy.load("en_core_web_sm")

# Step 2: Define Sample Sentences
test_sentences = [
    "The student reads a book in the library.",
    "Excited visitors ride the Space Mountain roller coaster at Disneyland."
]

print("=" * 75)
print("EXPERIMENT 3: DEPENDENCY PARSING WITH SPACY")
print("=" * 75)

for idx, sentence in enumerate(test_sentences, 1):
    doc = nlp(sentence)
    
    print(f"\nSentence #{idx}: \"{sentence}\"")
    print("-" * 75)
    print(f"{'Token':<16} | {'POS':<6} | {'Dependency':<12} | {'Head Word':<12} | {'Children / Dependents'}")
    print("-" * 75)
    
    for token in doc:
        children = [child.text for child in token.children]
        children_str = ", ".join(children) if children else "-"
        print(f"{token.text:<16} | {token.pos_:<6} | {token.dep_:<12} | {token.head.text:<12} | {children_str}")
        
    print("-" * 75)
    # Identify the root predicate
    root_token = [token for token in doc if token.dep_ == "ROOT"][0]
    print(f"Sentence Predicate (ROOT): '{root_token.text}' (POS: {root_token.pos_})")

# Step 3: Export SVG Dependency Visualization to file (instead of blocking port 5000)
svg_output_path = "dependency_tree.svg"
svg_content = displacy.render(nlp(test_sentences[0]), style="dep", jupyter=False)
with open(svg_output_path, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"\n[OK] Dependency visualization saved to: {svg_output_path}")
print("=" * 75)
print("[Summary] Dependency parsing directly links heads to dependents via directed relations.")
print("=" * 75)

