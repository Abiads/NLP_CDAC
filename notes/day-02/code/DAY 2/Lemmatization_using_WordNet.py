import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("DisneylandReviews.csv")

review = str(df['Review_Text'][0])

tokens = word_tokenize(review)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Original Review:")
print(review)

print("\nComparison of Stemming and Lemmatization")
print("-"*50)

print(f"{'Word':<15}{'Stem':<15}{'Lemma'}")

for word in tokens:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)

    print(f"{word:<15}{stem:<15}{lemma}")