from collections import defaultdict

# Training corpus
corpus = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "Machine learning is interesting"
]

# Transition table
transitions = defaultdict(lambda: defaultdict(int))

# Build Markov Model
for sentence in corpus:
    words = sentence.lower().split()

    for i in range(len(words) - 1):
        current_state = words[i]
        next_state = words[i + 1]

        transitions[current_state][next_state] += 1

# Calculate probabilities
probabilities = {}

for state, next_words in transitions.items():
    total = sum(next_words.values())

    probabilities[state] = {
        word: count / total
        for word, count in next_words.items()
    }

# Display transition probabilities
print("Transition Probabilities:\n")

for state, probs in probabilities.items():
    print(state, "->", probs)

# Predict next word
current_word = input("\nEnter current word: ").lower()

if current_word in probabilities:
    predicted_word = max(
        probabilities[current_word],
        key=probabilities[current_word].get
    )

    print("Predicted Next Word:", predicted_word)
else:
    print("Word not found in model.")