import nltk
nltk.download('gutenberg')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.collocations import (
    BigramCollocationFinder,
    BigramAssocMeasures
)
from nltk.corpus import gutenberg, wordnet as wn

# Syntagmatic: which words keep company?
words = [
    w.lower()
    for w in gutenberg.words("austen-emma.txt")
    if w.isalpha()
]

f = BigramCollocationFinder.from_words(words)

f.apply_freq_filter(5)

print("Top 10 PMI Bigrams")
print("-" * 40)

for bg in f.nbest(BigramAssocMeasures.pmi, 10):
    print(bg)

print("\nWordNet Synsets and Hypernyms")
print("-" * 40)

# Paradigmatic: which words can replace each other?
for syn in wn.synsets("car")[:3]:
    print("\nSynset:", syn.name())
    print("Lemmas:", syn.lemma_names())

    print("Hypernyms:")
    for h in syn.hypernyms():
        print("   ", h.name())