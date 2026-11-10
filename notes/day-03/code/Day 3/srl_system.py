
import spacy

# Step 1: Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Step 2: Define a simple rule-based SRL function
def semantic_role_labeling(sentence):

    doc = nlp(sentence)

    print("\nSentence:", sentence)
    print("-" * 50)

    for token in doc:

        # Identify the main predicate
        if token.pos_ == "VERB":

            print("\nPredicate / Action:", token.text)

            # Examine words related to the predicate
            for child in token.children:

                # Agent: subject performing the action
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    print("Agent:", child.text)

                # Patient: object affected by the action
                elif child.dep_ in ["dobj", "obj"]:
                    print("Patient:", child.text)

                # Instrument: tool used to perform the action
                elif child.dep_ == "prep" and child.text == "with":

                    for obj in child.children:
                        if obj.dep_ == "pobj":
                            print("Instrument:", obj.text)

# Step 3: Provide sample sentences
sentences = [
    "John writes a letter.",
    "Mary cuts the vegetables with a knife.",
    "The boy kicks the ball.",
    "The teacher explains the lesson."
]

# Step 4: Apply SRL to each sentence
for sentence in sentences:
    semantic_role_labeling(sentence)
