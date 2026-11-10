import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import zipfile

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Save the uploaded ZIP bytes to disk
if 'uploaded' in globals() and 'DisneylandReviews.csv.zip' in uploaded:
    with open("DisneylandReviews.csv.zip", "wb") as f:
        f.write(uploaded['DisneylandReviews.csv.zip'])

# Extract the ZIP archive
with zipfile.ZipFile("DisneylandReviews.csv.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# Read the extracted CSV file with proper encoding
df = pd.read_csv("DisneylandReviews.csv", encoding='latin-1')

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