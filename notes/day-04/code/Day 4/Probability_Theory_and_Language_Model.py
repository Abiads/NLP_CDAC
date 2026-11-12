import nltk
from nltk.util import ngrams
from collections import Counter

# Sample corpus
corpus = """
natural language processing is a part of artificial intelligence
natural language processing helps computers understand human language
artificial intelligence uses statistical models
"""

# Tokenize
words = corpus.lower().split()

# Generate N-grams
unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

# Count frequencies
unigram_count = Counter(unigrams)
bigram_count = Counter(bigrams)
trigram_count = Counter(trigrams)

total_words = len(words)

print("UNIGRAM PROBABILITIES")
for word, count in unigram_count.items():
    print(word[0], ":", round(count/total_words, 4))

# Sentence for probability calculation
sentence = "natural language processing"
tokens = sentence.lower().split()

# Unigram Probability
uni_prob = 1
for word in tokens:
    uni_prob *= unigram_count[(word,)] / total_words

print("\nSentence:", sentence)
print("Unigram Probability =", uni_prob)

# Bigram Probability
bi_prob = 1

for i in range(len(tokens)-1):
    bg = (tokens[i], tokens[i+1])

    if bg in bigram_count:
        bi_prob *= bigram_count[bg] / unigram_count[(tokens[i],)]
    else:
        bi_prob = 0

print("Bigram Probability =", bi_prob)

# Trigram Probability
tri_prob = 1

for i in range(len(tokens)-2):
    tg = (tokens[i], tokens[i+1], tokens[i+2])
    bg = (tokens[i], tokens[i+1])

    if tg in trigram_count:
        tri_prob *= trigram_count[tg] / bigram_count[bg]
    else:
        tri_prob = 0

print("Trigram Probability =", tri_prob)