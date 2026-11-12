
from collections import defaultdict, Counter

# STEP 1: Create a sample text corpus
corpus = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "I study artificial intelligence",
    "Machine learning is interesting"
]

# STEP 2: Tokenization
tokenized_corpus = []

for sentence in corpus:
    words = sentence.lower().split()
    tokenized_corpus.append(words)

print("TOKENIZED CORPUS:")
for sentence in tokenized_corpus:
    print(sentence)


# STEP 3: Count state transitions
transition_counts = defaultdict(Counter)

for sentence in tokenized_corpus:
    for i in range(len(sentence) - 1):
        current_state = sentence[i]
        next_state = sentence[i + 1]

        transition_counts[current_state][next_state] += 1


# STEP 4: Calculate transition probabilities
transition_probabilities = {}

for current_state, next_states in transition_counts.items():

    total = sum(next_states.values())

    transition_probabilities[current_state] = {}

    for next_state, count in next_states.items():

        probability = count / total

        transition_probabilities[current_state][next_state] = probability


# STEP 5: Display transition probabilities
print("\nSTATE TRANSITION PROBABILITIES:")

for current_state, next_states in transition_probabilities.items():

    print("\nCurrent State:", current_state)

    for next_state, probability in next_states.items():

        print(
            "Next State:", next_state,
            "Probability:", round(probability, 2)
        )


# STEP 6: Predict the next word
def predict_next_word(current_word):

    current_word = current_word.lower()

    if current_word not in transition_probabilities:
        print("\nNo transition information available for:", current_word)
        return

    next_states = transition_probabilities[current_word]

    predicted_word = max(next_states, key=next_states.get)

    print("\nCurrent Word:", current_word)
    print("Predicted Next Word:", predicted_word)
    print("Transition Probability:",
          round(next_states[predicted_word], 2))


# STEP 7: Test the Markov Model
predict_next_word("i")
predict_next_word("love")
predict_next_word("machine")
