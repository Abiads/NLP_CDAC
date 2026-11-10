import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("DisneylandReviews.csv")

review = str(df['Review_Text'][0])

print("Review:")
print(review)

doc = nlp(review)

print("\nNamed Entities:\n")

for ent in doc.ents:
    print(f"{ent.text:<20} --> {ent.label_}")