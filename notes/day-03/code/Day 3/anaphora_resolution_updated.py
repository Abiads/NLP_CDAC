import spacy

# Load standard pre-installed spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Rahul is a student.
He studies computer science.
Rahul likes Python.
He wants to become a software engineer.
"""

doc = nlp(text)

print("ANAPHORA AND COREFERENCE RESOLUTION")
print("----------------------------------")

# 1. Identify Potential Antecedents (Proper Nouns representing people)
print("\nPotential Antecedents (Proper Nouns / Entities):")
antecedents = []

# Check for PERSON entities
for ent in doc.ents:
    if ent.label_ == "PERSON" and ent.text not in antecedents:
        antecedents.append(ent.text)

# Fallback/Supplemental: Check for proper nouns that are likely people names
for token in doc:
    if token.pos_ == "PROPN" and token.text not in antecedents and token.text not in ["Python"]:
        antecedents.append(token.text)

for ant in antecedents:
    print(f"   - {ant}")

# 2. Rule-Based Coreference/Anaphora Matching
print("\nPronoun-Antecedent Resolution:")
current_antecedent = antecedents[0] if antecedents else "Unknown"

for sent in doc.sents:
    for token in sent:
        # If we hit a known antecedent in the text, update our current context tracking
        if token.text in antecedents:
            current_antecedent = token.text
        
        # Check if the token is a masculine pronoun
        if token.pos_ == "PRON" and token.text.lower() in ["he", "him", "his", "himself"]:
            print(f"   - Resolved Pronoun: '{token.text}' -> '{current_antecedent}' in sentence: \"{sent.text.strip()}\"")

print("\nCoreference Resolution Completed.")