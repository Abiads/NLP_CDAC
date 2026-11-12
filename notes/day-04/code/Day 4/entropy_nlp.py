import math
from collections import Counter

# Corpus 1
text1 = """
machine learning is a branch of artificial intelligence
machine learning helps computers learn from data
"""

# Corpus 2
text2 = """
deep learning is a subset of machine learning
deep learning uses neural networks
"""

# Tokenization
words1 = text1.lower().split()
words2 = text2.lower().split()

# Word frequencies
freq1 = Counter(words1)
freq2 = Counter(words2)

# Probabilities
total1 = sum(freq1.values())
total2 = sum(freq2.values())

prob1 = {word: count / total1 for word, count in freq1.items()}
prob2 = {word: count / total2 for word, count in freq2.items()}

# Entropy Calculation
entropy = 0

for word in prob1:
    p = prob1[word]
    entropy += -p * math.log2(p)

print("Entropy of Corpus 1 =", entropy)

# Cross Entropy Calculation
cross_entropy = 0

for word in prob1:
    p = prob1[word]
    q = prob2.get(word, 0.0001)  # smoothing
    cross_entropy += -p * math.log2(q)

print("Cross Entropy =", cross_entropy)