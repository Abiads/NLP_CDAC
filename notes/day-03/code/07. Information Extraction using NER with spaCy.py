"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 9: Information Extraction using Named Entity Recognition (NER)
File:   07. Information Extraction using NER with spaCy.py
=============================================================================

WHAT IS INFORMATION EXTRACTION (IE)?
------------------------------------
Information Extraction (IE) is the task of automatically transforming
unstructured natural language text into structured, relational knowledge
(entities, relations, and events).

WHAT IS NAMED ENTITY RECOGNITION (NER)?
---------------------------------------
Named Entity Recognition (NER) is a core subtask of Information Extraction.
It identifies the spans of text representing specific entities and classifies
them into predefined ontological categories:
  - PERSON   : People (e.g., 'Dr. A.P.J. Abdul Kalam', 'Bill Gates')
  - ORG      : Organizations, companies, agencies (e.g., 'ISRO', 'Microsoft')
  - GPE      : Geo-Political Entities - countries, cities, states (e.g., 'India', 'Bengaluru')
  - DATE     : Absolute or relative dates/periods (e.g., '1975', '15 August 2026')
  - MONEY    : Monetary values including units (e.g., '$5 million')
  - PERCENT  : Percentage values (e.g., '25%')
  - PRODUCT  : Physical goods, vehicles, software (e.g., 'iPhone', 'Space Mountain')
  - EVENT    : Named occurrences, conferences, sports (e.g., 'Olympic Games')

HOW SPACY NER WORKS:
--------------------
spaCy uses a transition-based parser with a Bloom filter / convolutional embedder
or transformer architecture that tags tokens with Biluo / BIO schemes:
  B (Begin), I (Inside), L (Last), U (Unit), O (Outside).
=============================================================================
"""

import spacy
from typing import List, Dict, Any

# Step 1: Load spaCy English pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[Error] spaCy model 'en_core_web_sm' is not installed.")
    print("Run: python -m spacy download en_core_web_sm")
    exit(1)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts named entities from a given text and annotates their labels,
    character spans, and descriptions.
    """
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "description": spacy.explain(ent.label_) or "Custom Entity"
        })
    return entities


def display_extracted_entities(text: str, entities: List[Dict[str, Any]]):
    """
    Prints input text and a cleanly formatted table of extracted entities.
    """
    print("-" * 80)
    print("Input Text:")
    print(f"  \"{text.strip()}\"")
    print("-" * 80)
    
    if not entities:
        print("  [No named entities recognized in text]")
        print()
        return
        
    print(f"  {'Entity Text':<30} | {'Type':<10} | {'Span':<12} | {'Description'}")
    print("  " + "-" * 76)
    for ent in entities:
        span_str = f"[{ent['start_char']}..{ent['end_char']}]"
        print(f"  {ent['text']:<30} | {ent['label']:<10} | {span_str:<12} | {ent['description']}")
    print()


def main():
    print("=" * 80)
    print("EXPERIMENT 9: INFORMATION EXTRACTION USING NER WITH SPACY")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & Information Extraction")
    print("=" * 80)

    # Test cases covering academic, corporate, calendar, monetary, and domain text
    test_cases = [
        "Dr. A.P.J. Abdul Kalam worked at ISRO in India.",
        "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque in 1975.",
        "The AI Summit conference will be held in Bengaluru on 15 August 2026.",
        "Alphabet invested $5 million in quantum computing research in Zurich.",
        "During our July vacation, we visited Disneyland Paris and rode Space Mountain with John."
    ]

    for sentence in test_cases:
        extracted = extract_entities(sentence)
        display_extracted_entities(sentence, extracted)

    # Entity Frequency Summary across all texts
    print("=" * 80)
    print("GLOBAL ENTITY TYPE SUMMARY ACROSS CORPUS:")
    print("=" * 80)
    
    all_ents = []
    for s in test_cases:
        all_ents.extend(extract_entities(s))
        
    type_counts = {}
    for e in all_ents:
        t = e["label"]
        type_counts[t] = type_counts.get(t, 0) + 1
        
    print(f"  {'Entity Label':<15} | {'Count':<8} | {'Description'}")
    print("  " + "-" * 60)
    for label, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<15} | {count:<8} | {spacy.explain(label)}")
        
    print("=" * 80)
    print("[Summary] NER identifies key informational anchors (Who, What, Where, When)")
    print("enabling downstream Relation Extraction and Knowledge Graph construction.")
    print("=" * 80)


if __name__ == "__main__":
    main()

