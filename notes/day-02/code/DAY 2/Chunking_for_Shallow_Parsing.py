
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Input sentence
sentence = "The beautiful girl is reading a book in the library"

# Step 1: Tokenization
tokens = word_tokenize(sentence)

print("Original Sentence:")
print(sentence)

print("\nTokens:")
print(tokens)

# Step 2: POS Tagging
tagged_words = pos_tag(tokens)

print("\nPOS Tagged Words:")
print(tagged_words)

# Step 3: Define Chunking Grammar
grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
    VP: {<VB.*><NP>?}
    PP: {<IN><NP>}
"""

# Step 4: Create Chunk Parser
chunk_parser = RegexpParser(grammar)

# Step 5: Perform Chunking
chunk_tree = chunk_parser.parse(tagged_words)

# Step 6: Display Chunked Output
print("\nChunked Output:")
print(chunk_tree)

# Step 7: Display Tree
chunk_tree.pretty_print()
