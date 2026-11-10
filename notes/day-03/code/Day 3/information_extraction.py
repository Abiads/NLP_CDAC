
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample sentences
text = """
Dr. A.P.J. Abdul Kalam worked at ISRO in India.
Microsoft was founded by Bill Gates in 1975.
The conference will be held in Bengaluru on 15 August 2026.
The company invested $5 million in the project.
"""

# Process the text
doc = nlp(text)

# Display the original text
print("INFORMATION EXTRACTION USING NER")
print("--------------------------------")
print("Input Text:")
print(text)

# Extract named entities
print("\nExtracted Named Entities:")
print("--------------------------------")

for ent in doc.ents:
    print("Entity:", ent.text)
    print("Entity Type:", ent.label_)
    print("Description:", spacy.explain(ent.label_))
    print("--------------------------------")
