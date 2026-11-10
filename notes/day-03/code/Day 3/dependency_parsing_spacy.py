import spacy
from spacy import displacy

# Step 1: Load the English language model
nlp = spacy.load("en_core_web_sm")

# Step 2: Provide a sentence
sentence = "The student reads a book."

# Step 3: Process the sentence
doc = nlp(sentence)

# Step 4: Display dependency relationships
print("Dependency Parsing Results")
print("-----------------------")

for token in doc:
    print(
        token.text,
        " | POS:", token.pos_,
        " | Dependency:", token.dep_,
        " | Head:", token.head.text
    )

# Step 5: Visualize dependency tree
displacy.serve(
    doc,
    style="dep",
    port=5000
)
