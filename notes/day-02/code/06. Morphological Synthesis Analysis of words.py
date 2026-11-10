"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Morphological Analysis and Morphological Synthesis
File:   06_morphological_analysis_and_synthesis.py
=============================================================================

1. WHAT IS MORPHOLOGICAL ANALYSIS?
----------------------------------
Morphological Analysis is the process of taking a surface word (the word as
it appears in text) and decomposing it into:
  - Its base dictionary root (lemma / stem)
  - Its grammatical features (e.g., Number, Tense, Person, Aspect, Degree)

Example:
  Input:  "children"
  Output: Root = "child", Part-of-Speech = NOUN, Number = Plural

  Input:  "played"
  Output: Root = "play",  Part-of-Speech = VERB, Tense = Past


2. WHAT IS MORPHOLOGICAL SYNTHESIS (GENERATION)?
------------------------------------------------
Morphological Synthesis is the EXACT REVERSE of Morphological Analysis.
It takes a root lemma and a set of desired grammatical features, and generates
the correct inflected surface word form.

Example:
  Input:  Root = "child", Features = {POS: "NOUN", Number: "Plural"}
  Output: "children"

  Input:  Root = "play",  Features = {POS: "VERB", Tense: "Past"}
  Output: "played"
=============================================================================
"""

import spacy

# Load spaCy English language model for morphological analysis
nlp = spacy.load("en_core_web_sm")

# ===========================================================================
# PART 1: MORPHOLOGICAL ANALYSIS (Surface Word -> Root + Features)
# ===========================================================================
print("=" * 70)
print("PART 1: MORPHOLOGICAL ANALYSIS (Decomposing Words)")
print("=" * 70)

sample_sentence = "The children played with their newer toys happily."
doc = nlp(sample_sentence)

print(f"Sample Sentence: '{sample_sentence}'\n")
print(f"{'Surface Token':<15} {'Root Lemma':<12} {'POS':<8} {'Morphological Features'}")
print("-" * 70)

for token in doc:
    # Filter out punctuation for clean tabular display
    if not token.is_punct:
        morph_features = str(token.morph) if str(token.morph) else "None"
        print(f"{token.text:<15} {token.lemma_:<12} {token.pos_:<8} {morph_features}")


# ===========================================================================
# PART 2: MORPHOLOGICAL SYNTHESIS (Root + Features -> Surface Word)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2: MORPHOLOGICAL SYNTHESIS (Generating Word Forms)")
print("=" * 70)

# Irregular dictionaries for accurate English inflection
irregular_plurals = {
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
    "person": "people",
    "man": "men",
    "woman": "women"
}

irregular_verbs = {
    "run": {"Past": "ran", "Participle": "run", "Progressive": "running"},
    "go": {"Past": "went", "Participle": "gone", "Progressive": "going"},
    "eat": {"Past": "ate", "Participle": "eaten", "Progressive": "eating"},
    "see": {"Past": "saw", "Participle": "seen", "Progressive": "seeing"},
    "write": {"Past": "wrote", "Participle": "written", "Progressive": "writing"}
}

def synthesize_noun(root, number="Plural"):
    """Generates the singular or plural form of a noun."""
    if number == "Singular":
        return root
    
    # Check irregular plurals first
    if root.lower() in irregular_plurals:
        return irregular_plurals[root.lower()]
    
    # Regular pluralization rules
    if root.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return root + "es"
    elif root.endswith('y') and len(root) > 1 and root[-2] not in "aeiou":
        return root[:-1] + "ies"
    else:
        return root + "s"

def synthesize_verb(root, tense="Past", aspect="Simple", person="3rd_Singular"):
    """Generates the appropriate inflected verb form based on tense and aspect."""
    # Check irregular verbs
    if root.lower() in irregular_verbs:
        if aspect == "Progressive":
            return irregular_verbs[root.lower()]["Progressive"]
        if tense == "Past":
            return irregular_verbs[root.lower()]["Past"]
    
    # Continuous / Progressive (-ing)
    if aspect == "Progressive":
        if root.endswith('e') and not root.endswith(('ee', 'oe')):
            return root[:-1] + "ing"
        return root + "ing"
    
    # Past Tense (-ed)
    if tense == "Past":
        if root.endswith('e'):
            return root + "d"
        elif root.endswith('y') and len(root) > 1 and root[-2] not in "aeiou":
            return root[:-1] + "ied"
        else:
            return root + "ed"
            
    # Present 3rd person singular (-s / -es)
    if person == "3rd_Singular":
        if root.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return root + "es"
        elif root.endswith('y') and len(root) > 1 and root[-2] not in "aeiou":
            return root[:-1] + "ies"
        else:
            return root + "s"
            
    return root

def synthesize_adjective(root, degree="Comparative"):
    """Generates comparative (-er) or superlative (-est) adjective forms."""
    if degree == "Comparative":
        if root == "good": return "better"
        if root == "bad": return "worse"
        if root.endswith('e'): return root + "r"
        if root.endswith('y'): return root[:-1] + "ier"
        return root + "er"
    elif degree == "Superlative":
        if root == "good": return "best"
        if root == "bad": return "worst"
        if root.endswith('e'): return root + "st"
        if root.endswith('y'): return root[:-1] + "iest"
        return root + "est"
    return root

# Demonstration of Morphological Synthesis
synthesis_tasks = [
    # Nouns
    ("child", "NOUN", {"Number": "Plural"}),
    ("box", "NOUN", {"Number": "Plural"}),
    ("city", "NOUN", {"Number": "Plural"}),
    
    # Verbs
    ("play", "VERB", {"Tense": "Past"}),
    ("write", "VERB", {"Tense": "Past"}),
    ("study", "VERB", {"Tense": "Past"}),
    ("run", "VERB", {"Aspect": "Progressive"}),
    ("teach", "VERB", {"Person": "3rd_Singular"}),
    
    # Adjectives
    ("fast", "ADJ", {"Degree": "Comparative"}),
    ("happy", "ADJ", {"Degree": "Comparative"}),
    ("good", "ADJ", {"Degree": "Superlative"})
]

print(f"{'Root Lemma':<12} {'Target POS':<10} {'Requested Features':<30} --> {'Synthesized Word'}")
print("-" * 75)

for root, pos, feats in synthesis_tasks:
    if pos == "NOUN":
        output = synthesize_noun(root, feats.get("Number", "Plural"))
    elif pos == "VERB":
        output = synthesize_verb(root, feats.get("Tense", "Present"), feats.get("Aspect", "Simple"), feats.get("Person", "Default"))
    elif pos == "ADJ":
        output = synthesize_adjective(root, feats.get("Degree", "Comparative"))
    else:
        output = root
        
    feats_str = ", ".join(f"{k}={v}" for k, v in feats.items())
    print(f"{root:<12} {pos:<10} {feats_str:<30} --> {output}")


# ===========================================================================
# PART 3: THE PARADIGM EXAMPLE (Session 3 Slide 6 Reference)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 3: INFLECTIONAL PARADIGM EXAMPLE (Hindi Noun Paradigm)")
print("=" * 70)
print("As presented in Ma'am's Slide 6: Hindi Masculine Nouns in -aa")
print("Root: 'ladka' (boy)\n")

hindi_paradigm = [
    ("Direct Singular", "ladka", "larka"),
    ("Direct Plural",   "ladke", "larke"),
    ("Oblique Singular","ladke", "larke ko"),
    ("Oblique Plural",  "ladkon", "larkon ko")
]

print(f"{'Grammatical Form':<20} | {'Inflected Surface Word':<25} | {'Usage in Sentence'}")
print("-" * 70)
for form, word, example in hindi_paradigm:
    print(f"{form:<20} | {word:<25} | {example}")

print("=" * 70)
print("[Summary] Analysis converts Word -> Root + Features.")
print("[Summary] Synthesis converts Root + Features -> Word.")
print("=" * 70)
