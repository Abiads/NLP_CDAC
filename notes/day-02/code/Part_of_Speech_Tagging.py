import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import zipfile

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

import os

# Save the uploaded ZIP bytes to disk (if in Colab environment)
if 'uploaded' in globals() and 'DisneylandReviews.csv.zip' in uploaded:
    with open("DisneylandReviews.csv.zip", "wb") as f:
        f.write(uploaded['DisneylandReviews.csv.zip'])

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

# Extract the ZIP archive only if CSV is not already present
if not os.path.exists(csv_path) and os.path.exists("DisneylandReviews.csv.zip"):
    with zipfile.ZipFile("DisneylandReviews.csv.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    csv_path = "DisneylandReviews.csv"

# Read the extracted CSV file with proper encoding
df = pd.read_csv(csv_path, encoding='latin-1')

review = str(df['Review_Text'][0])

print("Original Review:")
print(review)

tokens = word_tokenize(review)

print("\nTokens:")
print(tokens)

pos_tags = pos_tag(tokens)

print("\nPOS Tagging Results")
print("--------------------")

for word, tag in pos_tags:
    print(f"{word:<15} --> {tag}")