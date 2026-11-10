import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
# Input sentence
sentence = "The New York City tourist visited the theme park"
print("Original Sentence:")
print(sentence)
# Step 1: Tokenization
tokens = word_tokenize(sentence)
print("\nTokens:")
print(tokens)
# Step 2: Define Multi-Word Expressions
mwe = [
    ("New", "York", "City"), ("theme", "park")
]
# Step 3: Create MWE Tokenizer
mwe_tokenizer = MWETokenizer(mwe)
# Step 4: Identify Multi-Word Expressions
mwe_tokens = mwe_tokenizer.tokenize(tokens)
# Step 5: Display Output
print("\nMulti-Word Expression Identification:")
print(mwe_tokens)
# Step 6: Display extracted MWEs
print("\nExtracted Multi-Word Expressions:")
for word in mwe_tokens:
    if "_" in word:
        print(word)