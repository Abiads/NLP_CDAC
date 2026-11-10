import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer

nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv("DisneylandReviews.csv")

review = str(df['Review_Text'][0])

print("Original Review:")
print(review)

tokens = word_tokenize(review.lower())

mwes = [
    ('theme', 'park'),
    ('customer', 'service'),
    ('roller', 'coaster'),
    ('food', 'court'),
    ('waiting', 'time')
]

tokenizer = MWETokenizer(mwes, separator='_')

mwe_tokens = tokenizer.tokenize(tokens)

print("\nAfter MWE Identification:\n")
print(mwe_tokens)

print("\nDetected MWEs:\n")

for token in mwe_tokens:
    if '_' in token:
        print(token)