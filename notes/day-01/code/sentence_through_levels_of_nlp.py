import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(
    "The children in Pune didn't eat "
    "the mangoes because they were raw."
)

for t in doc:
    print(
        t.text,
        t.lemma_,
        t.morph,
        t.pos_,
        t.dep_,
        t.head.text
    )

print("\nNamed Entities:")
print(doc.ents)