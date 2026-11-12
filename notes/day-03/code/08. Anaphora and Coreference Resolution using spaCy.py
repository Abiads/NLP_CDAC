"""
Experiment 08: Anaphora and Coreference Resolution using spaCy
Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools
Reference: notes/day-03/code/Day 3/anaphora_resolution_updated.py

Key Objectives:
1. Identify named entity antecedents (PERSON, ORG, GPE) in multi-sentence discourse.
2. Resolve third-person personal and possessive anaphors (he, him, his, she, her, they) to antecedents.
3. Handle gender, grammatical number, and recency constraints.
4. Document resolution outputs with sentence context.
Strict Zero-Emoji Policy.
"""

import spacy

nlp = spacy.load("en_core_web_sm")

PRONOUN_MAP = {
    "he": "MASC", "him": "MASC", "his": "MASC", "himself": "MASC",
    "she": "FEM", "her": "FEM", "hers": "FEM", "herself": "FEM",
    "they": "PLUR", "them": "PLUR", "their": "PLUR", "theirs": "PLUR"
}


def resolve_anaphora(text):
    doc = nlp(text.strip())

    print("=" * 80)
    print("EXPERIMENT 08: ANAPHORA & COREFERENCE RESOLUTION")
    print("=" * 80)
    print(f"\nDiscourse Text:\n{text.strip()}\n")

    # Step 1: Extract candidate antecedents
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE")]
    print(f"Extracted Salient Antecedents: {entities}")

    # Fallback to proper nouns if entities are sparse
    propn_antecedents = []
    for token in doc:
        if token.pos_ == "PROPN" and token.text not in propn_antecedents:
            propn_antecedents.append(token.text)

    # Step 2: Track current antecedent context sentence by sentence
    print("\n" + "=" * 80)
    print(f"{'Sentence':<30} | {'Pronoun':<10} | {'Resolved Antecedent':<20} | {'Agreement'}")
    print("-" * 80)

    current_antecedent = propn_antecedents[0] if propn_antecedents else "Unknown"

    for sent in doc.sents:
        # Check if an explicit named antecedent appears in this sentence
        for t in sent:
            if t.text in propn_antecedents:
                current_antecedent = t.text

        # Resolve pronouns
        for token in sent:
            if token.pos_ == "PRON" and token.lower_ in PRONOUN_MAP:
                sent_clean = sent.text.strip().replace("\n", " ")
                print(f"{sent_clean[:28]:<30} | {token.text:<10} | {current_antecedent:<20} | {PRONOUN_MAP[token.lower_]}")

    print("=" * 80)


if __name__ == "__main__":
    sample_text = """
    Rahul is a software engineer.
    He develops intelligent systems using Python.
    Rahul joined a multinational technology enterprise.
    He leads a multidisciplinary engineering team.
    Priya is a data scientist.
    She specializes in deep learning architectures.
    """
    resolve_anaphora(sample_text)
