
from collections import defaultdict, Counter

# STEP 1: Training data
text = """
the cat is sitting on the mat
the cat is playing with a ball
the dog is playing in the garden
the dog is sitting near the door
the cat is sleeping on the sofa
"""

# STEP 2: Tokenization
words = text.lower().split()

# STEP 3: Get N value from user
n = int(input("Enter N value (1, 2, 3 or 4): "))

# STEP 4: Create N-gram model
ngram_model = defaultdict(Counter)

for i in range(len(words) - n + 1):

    ngram = tuple(words[i:i+n])

    # Previous n-1 words form the context
    context = ngram[:-1]

    # Last word is the predicted word
    next_word = ngram[-1]

    ngram_model[context][next_word] += 1

# STEP 5: Display generated N-grams
print("\nGenerated N-grams:")

for context, predictions in ngram_model.items():
    for word, count in predictions.items():
        print(context, "->", word, ":", count)

# STEP 6: Get input sequence
user_input = input("\nEnter preceding words: ").lower().split()

# STEP 7: Extract context
context = tuple(user_input[-(n-1):]) if n > 1 else ()

# STEP 8: Predict next word
print("\nPrediction:")

if context in ngram_model:

    prediction = ngram_model[context].most_common(1)

    print("Input context:", context)
    print("Predicted next word:", prediction[0][0])

else:
    print("No matching N-gram found.")
