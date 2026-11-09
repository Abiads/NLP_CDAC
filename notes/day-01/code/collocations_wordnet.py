import nltk
from nltk.collocations import (
    BigramCollocationFinder,
    BigramAssocMeasures
)
from nltk.corpus import gutenberg, wordnet as wn

# --------------------------------------------------
# Task 1: Collocations in Emma (PMI)
# --------------------------------------------------

words = [
    w.lower()
    for w in gutenberg.words("austen-emma.txt")
    if w.isalpha()
]

finder = BigramCollocationFinder.from_words(words)

# Ignore rare bigrams
finder.apply_freq_filter(5)

print("Task 1: Collocations in Emma (PMI)")
print("-" * 40)

for i, bg in enumerate(
        finder.nbest(BigramAssocMeasures.pmi, 10), start=1):
    print(f"{i}. {' '.join(bg)}")

# --------------------------------------------------
# Task 2: WordNet senses of car
# --------------------------------------------------

print("\nTask 2: WordNet Senses of 'car'")
print("-" * 40)

for syn in wn.synsets("car")[:3]:

    print("\n", syn.name())

    print("Synonyms:")
    print(", ".join(syn.lemma_names()))

    print("Hypernyms:")
    for h in syn.hypernyms():
        print("   ", h.name())