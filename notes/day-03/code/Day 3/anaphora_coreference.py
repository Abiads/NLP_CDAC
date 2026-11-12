import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Rahul is a student.
He studies computer science.
Rahul likes Python.
He wants to become a software engineer.
"""

# Process the text
doc = nlp(text)

print("ANAPHORA AND COREFERENCE RESOLUTION")
print("----------------------------------")

# Identify named entities
entities = []

for ent in doc.ents:
    if ent.label_ == "PERSON":
        entities.append(ent.text)

print("\nNamed Entities:")
for entity in entities:
    print(entity)

# Identify pronouns and their antecedents
print("\nPronouns and their Antecedents:")

antecedent = None

for token in doc:

    # Identify person names
    if token.ent_type_ == "PERSON":
        antecedent = token.text

    # Identify personal pronouns
    if token.pos_ == "PRON":

        if token.text.lower() in ["he", "him", "his"]:
            print(token.text, "->", antecedent)

        elif token.text.lower() in ["she", "her", "hers"]:
            print(token.text, "->", antecedent)

        elif token.text.lower() in ["they", "them", "their"]:
            print(token.text, "->", antecedent)

print("\nCoreference Resolution Completed.")