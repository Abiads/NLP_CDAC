import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Locate DisneylandReviews.csv robustly from any directory
csv_path = 'DisneylandReviews.csv'
if not os.path.exists(csv_path):
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', 'DisneylandReviews.csv'),
        os.path.join('notes', 'day-02', 'DisneylandReviews.csv'),
        os.path.join('assignments', 'assignment-02', 'DisneylandReviews.csv')
    ]:
        if os.path.exists(candidate):
            csv_path = candidate
            break

# Added encoding='latin1' to handle special non-UTF-8 characters in the CSV file
df = pd.read_csv(csv_path, encoding='latin1')

review = str(df['Review_Text'][0])

print("Original Review:")
print(review)

print("\nSentence Tokenization")
sentences = sent_tokenize(review)
for s in sentences:
    print(s)

print("\nWord Tokenization")
words = word_tokenize(review)
print(words)

print("\nStatistics")
print("Number of Sentences:", len(sentences))
print("Number of Words:", len(words))

