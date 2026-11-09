import spacy

nlp = spacy.load("en_core_web_sm")

sentences = [
    "Visiting aunts can be a nuisance.",
    "She went to the bank.",
    "I saw the man with the telescope.",
    "Can you pass the salt?"
]

for s in sentences:
    print("Sentence:", s)
    print("-" * 40)

    for tok in nlp(s):
        print(tok.text, tok.dep_, tok.head.text)

    print()