import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(
    "The children in Pune didn't eat "
    "the mangoes because they were raw."
)

print(f"{'TOKEN':<12} {'LEMMA':<12} {'MORPH':<25} {'POS':<8} {'DEP':<10} {'HEAD'}")
print("-" * 80)

for t in doc:
    print(
        f"{t.text:<12} "
        f"{t.lemma_:<12} "
        f"{str(t.morph):<25} "
        f"{t.pos_:<8} "
        f"{t.dep_:<10} "
        f"{t.head.text}"
    )

print("\nEntities:")
for ent in doc.ents:
    print(ent.text, ent.label_)