import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv("DisneylandReviews.csv")

review = str(df['Review_Text'][0])

print("Original Review:\n")
print(review)

tokens = word_tokenize(review)

ps = PorterStemmer()

print("\nStemmed Words\n")

for word in tokens:
    stem_word = ps.stem(word)
    print(f"{word:<15} --> {stem_word}")