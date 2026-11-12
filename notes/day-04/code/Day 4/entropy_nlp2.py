
from collections import Counter
import math

# STEP 1: Create two text corpora

# Actual text corpus
corpus_P = [
    "data science is useful",
    "data science is interesting",
    "machine learning is useful"
]

# Predicted/model text corpus
corpus_Q = [
    "data science is useful",
    "machine learning is useful",
    "data science is useful",
    "machine learning is interesting"
]


# STEP 2: Tokenization and word counting

words_P = []

for sentence in corpus_P:
    words_P.extend(sentence.lower().split())

words_Q = []

for sentence in corpus_Q:
    words_Q.extend(sentence.lower().split())


count_P = Counter(words_P)
count_Q = Counter(words_Q)

total_P = len(words_P)
total_Q = len(words_Q)


# STEP 3: Calculate word probabilities

vocabulary = sorted(set(words_P) | set(words_Q))

P = {}
Q = {}

for word in vocabulary:
    P[word] = count_P[word] / total_P
    Q[word] = count_Q[word] / total_Q


# STEP 4: Display word probabilities

print("WORD PROBABILITY DISTRIBUTIONS")

print("\nWord\t\tP (Actual)\tQ (Model)")

for word in vocabulary:
    print(
        word, "\t\t",
        round(P[word], 4), "\t\t",
        round(Q[word], 4)
    )


# STEP 5: Calculate Entropy

entropy = 0

for word in vocabulary:
    if P[word] > 0:
        entropy -= P[word] * math.log2(P[word])


# STEP 6: Calculate Cross-Entropy

cross_entropy = 0

for word in vocabulary:
    if P[word] > 0:

        if Q[word] == 0:
            cross_entropy = float("inf")
            break

        cross_entropy -= P[word] * math.log2(Q[word])


# STEP 7: Display results

print("\nENTROPY:", round(entropy, 4), "bits/word")

print("CROSS-ENTROPY:",
      round(cross_entropy, 4), "bits/word")
