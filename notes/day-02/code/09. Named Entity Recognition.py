"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 2 / Session 3
Topic:  Experiment 9: Named Entity Recognition (NER)
File:   09. Named Entity Recognition.py
=============================================================================

WHAT IS NAMED ENTITY RECOGNITION (NER)?
---------------------------------------
Named Entity Recognition (NER) is a core Information Extraction technique
in Natural Language Processing that identifies and classifies key elements
in unstructured text into predefined categories such as:
  - PERSON       : People, individuals (e.g., Walt Disney, Mickey Mouse)
  - GPE / LOC    : Geo-Political Entities or Locations (e.g., Paris, California)
  - ORG          : Organizations, companies (e.g., The Walt Disney Company)
  - DATE / TIME  : Calendar dates, times (e.g., July 17, 1955, morning)
  - MONEY        : Monetary values (e.g., $150, 50 euros)
  - FAC          : Buildings, airports, facilities (e.g., Disneyland Park)

THE IOB / BIO TAGGING SCHEME:
-----------------------------
NER uses BIO tagging to mark multi-token entities:
  - B- (Begin)  : First token of a named entity (e.g., B-PERSON)
  - I- (Inside) : Subsequent tokens of the same entity (e.g., I-PERSON)
  - O  (Outside): Tokens that are not part of any named entity

TOOLS COVERED IN THIS LAB:
--------------------------
1. NLTK ne_chunk (Maximum Entropy Classifier + POS Tags)
2. spaCy en_core_web_sm (Trained Transition-Based Neural NER Model)
3. Real Dataset: Entity extraction from Disneyland Reviews
=============================================================================
"""

import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Download required NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

# ---------------------------------------------------------------------------
# Sample Sentence
# ---------------------------------------------------------------------------
sample_text = (
    "Walt Disney officially opened Disneyland in Anaheim, California on July 17, 1955. "
    "Today, millions of tourists visit Disneyland Paris and pay over 100 dollars per ticket."
)

print("=" * 75)
print("EXPERIMENT 9: NAMED ENTITY RECOGNITION (NER)")
print("=" * 75)
print(f"Input Text:\n\"{sample_text}\"\n")

# ===========================================================================
# Part 1: NLTK NER using ne_chunk
# ===========================================================================
print("=" * 75)
print("1. NLTK NAMED ENTITY RECOGNITION (ne_chunk)")
print("=" * 75)

# Step 1: Tokenize
tokens = word_tokenize(sample_text)

# Step 2: Part-of-Speech Tagging
tagged_tokens = pos_tag(tokens)

# Step 3: NLTK Named Entity Chunker
# binary=False preserves distinct category labels (PERSON, GPE, ORGANIZATION)
# binary=True simply marks tokens as NE (Named Entity) or not
nltk_ner_tree = ne_chunk(tagged_tokens, binary=False)

print("NLTK Parsed Entity Tree (Truncated display):")
for subtree in nltk_ner_tree:
    if isinstance(subtree, nltk.Tree):
        entity_name = " ".join(leaf[0] for leaf in subtree.leaves())
        entity_label = subtree.label()
        print(f"  [Entity Found] {entity_name:<25} --> Label: {entity_label}")

# ===========================================================================
# Part 2: BIO / IOB Tagging Scheme (tree2conlltags)
# ===========================================================================
print("\n" + "=" * 75)
print("2. IOB / BIO TAGGING SCHEME (CoNLL Format)")
print("=" * 75)
print("IOB format maps every word into (Word, POS_Tag, BIO_Chunk_Tag):")
print(f"{'Token':<18} | {'POS Tag':<10} | {'IOB Tag'}")
print("-" * 45)

iob_tags = tree2conlltags(nltk_ner_tree)
for word, pos, iob in iob_tags[:15]:  # Display first 15 tokens
    print(f"{word:<18} | {pos:<10} | {iob}")

# ===========================================================================
# Part 3: Advanced NER using spaCy
# ===========================================================================
print("\n" + "=" * 75)
print("3. ADVANCED NER USING SPACY (en_core_web_sm)")
print("=" * 75)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sample_text)

    print(f"{'Entity Text':<25} | {'Label':<12} | {'Description'}")
    print("-" * 75)
    for ent in doc.ents:
        # spacy.explain() gives human-readable explanation of tag
        explanation = spacy.explain(ent.label_) or "Custom Entity"
        print(f"{ent.text:<25} | {ent.label_:<12} | {explanation}")

except ImportError:
    print("[Note] spaCy is not installed in current environment. NLTK covered above.")
except Exception as e:
    print(f"[Note] spaCy model load note: {e}")

# ===========================================================================
# Part 4: Real Dataset Application on Disneyland Reviews
# ===========================================================================
print("\n" + "=" * 75)
print("4. REAL DATASET APPLICATION (Disneyland Reviews)")
print("=" * 75)

csv_path = "DisneylandReviews.csv"
if not os.path.exists(csv_path):
    # Check parent directory fallback
    parent_path = os.path.join("..", "DisneylandReviews.csv")
    if os.path.exists(parent_path):
        csv_path = parent_path

if os.path.exists(csv_path):
    print(f"Loading sample reviews from: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin-1', nrows=5)
    
    # Process the first 3 reviews for entity extraction
    for idx, row in df.head(3).iterrows():
        review_id = row.get('Review_ID', idx + 1)
        branch = row.get('Branch', 'Unknown')
        text = str(row.get('Review_Text', ''))[:200] + "..."
        
        print(f"\n--- Review #{review_id} (Branch: {branch}) ---")
        print(f"Text Snippet: \"{text}\"")
        
        # Extract entities using spaCy if available
        try:
            review_doc = nlp(text)
            entities_found = [(e.text, e.label_) for e in review_doc.ents]
            if entities_found:
                print("Entities Found:")
                for ent_text, ent_label in entities_found:
                    print(f"   * {ent_text:<20} [{ent_label}]")
            else:
                print("   * No high-confidence entities found in snippet.")
        except Exception:
            # Fallback to NLTK ne_chunk
            r_tokens = word_tokenize(text)
            r_tree = ne_chunk(pos_tag(r_tokens))
            for subtree in r_tree:
                if isinstance(subtree, nltk.Tree):
                    ent_str = " ".join(leaf[0] for leaf in subtree.leaves())
                    print(f"   * {ent_str:<20} [{subtree.label()}]")
else:
    print("[Note] DisneylandReviews.csv not found in current folder.")

print("\n" + "=" * 75)
print("[Summary] NER successfully extracts domain entities (Persons, Places, Dates, Money).")
print("=" * 75)

