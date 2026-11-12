
from collections import Counter

# STEP 1: Create a sample text corpus
corpus = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "I study artificial intelligence",
    "Machine learning is interesting"
]

# STEP 2: Convert sentences into lowercase words
tokenized_corpus = []

for sentence in corpus:
    words = sentence.lower().split()
    tokenized_corpus.append(words)

print("Tokenized Corpus:")
for sentence in tokenized_corpus:
    print(sentence)

# STEP 3: Count individual words (Unigram)
unigram_counts = Counter()

for sentence in tokenized_corpus:
    unigram_counts.update(sentence)

total_words = sum(unigram_counts.values())

# STEP 4: Count consecutive two-word sequences (Bigram)
bigram_counts = Counter()
for sentence in tokenized_corpus:
    for i in range(len(sentence) - 1):
        bigram = (sentence[i], sentence[i+1])
        bigram_counts[bigram] += 1

# STEP 5: Count consecutive three-word sequences (Trigram)
trigram_counts = Counter()
for sentence in tokenized_corpus:
    for i in range(len(sentence) - 2):
        trigram = (
            sentence[i],
            sentence[i+1],
            sentence[i+2]
        )
        trigram_counts[trigram] += 1

# STEP 6: Get probability of a sentence
test_sentence = "I love machine learning"
words = test_sentence.lower().split()

# UNIGRAM PROBABILITY
unigram_probability = 1.0

for word in words:
    probability = unigram_counts[word] / total_words
    unigram_probability *= probability

# BIGRAM PROBABILITY
bigram_probability = 1.0
bigram_words = ["<s>"] + words

for i in range(len(bigram_words) - 1):
    first = bigram_words[i]
    second = bigram_words[i+1]

    numerator = bigram_counts[(first, second)]

    if first == "<s>":
        denominator = len(tokenized_corpus)
    else:
        denominator = unigram_counts[first]

    probability = numerator / denominator
    bigram_probability *= probability

# TRIGRAM PROBABILITY
trigram_probability = 1.0
trigram_words = ["<s>", "<s>"] + words

for i in range(len(trigram_words) - 2):
    first = trigram_words[i]
    second = trigram_words[i+1]
    third = trigram_words[i+2]

    numerator = trigram_counts[(first, second, third)]

    if (first, second) == ("<s>", "<s>"):
        denominator = len(tokenized_corpus)
    elif first == "<s>":
        denominator = sum(
            1 for sentence in tokenized_corpus
            if sentence[0] == second
        )
    else:
        denominator = bigram_counts[(first, second)]

    probability = numerator / denominator
    trigram_probability *= probability

# STEP 7: Display results
print("\nWord Counts:")
print(unigram_counts)

print("\nBigram Counts:")
print(bigram_counts)

print("\nTrigram Counts:")
print(trigram_counts)

print("\nSentence:", test_sentence)

print("\nUNIGRAM PROBABILITY:", unigram_probability)
print("BIGRAM PROBABILITY:", bigram_probability)
print("TRIGRAM PROBABILITY:", trigram_probability)
