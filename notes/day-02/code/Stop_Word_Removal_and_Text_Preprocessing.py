import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Creating a synthetic dataframe with sample Disneyland reviews to ensure the cell runs reliably
data = {
    'Review_Text': [
        "Had a wonderful time at Disneyland! The rides were amazing and the staff was extremely friendly, though the lines were quite long.",
        "Disneyland was okay, but it was way too crowded and expensive. Not worth the long wait times."
    ]
}
df = pd.DataFrame(data)

review = str(df['Review_Text'][0])

print("Original Review:")
print(review)

review = review.lower()

tokens = word_tokenize(review)

tokens_no_punct = [word for word in tokens
                   if word not in string.punctuation]

stop_words = set(stopwords.words('english'))

filtered_words = [word for word in tokens_no_punct
                  if word not in stop_words]

print("\nTokens:")
print(tokens)

print("\nAfter Removing Punctuation:")
print(tokens_no_punct)

print("\nAfter Stop Word Removal:")
print(filtered_words)

print("\nStatistics")
print("Words Before Cleaning:", len(tokens))
print("Words After Cleaning:", len(filtered_words))