from collections import defaultdict

corpus = """
i love machine learning
i love deep learning
i study machine learning
machine learning is interesting
"""

words = corpus.lower().split()

# Build bigrams
bigrams = defaultdict(list)

for i in range(len(words)-1):
    current_word = words[i]
    next_word = words[i+1]
    bigrams[current_word].append(next_word)

# Predict next word
input_word = input("Enter a word: ")

if input_word in bigrams:
    prediction = max(set(bigrams[input_word]),
                     key=bigrams[input_word].count)
    print("Predicted next word:", prediction)
else:
    print("Word not found in corpus")