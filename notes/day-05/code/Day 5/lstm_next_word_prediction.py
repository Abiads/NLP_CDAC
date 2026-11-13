
# NEXT WORD PREDICTION USING LSTM NETWORKS

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Embedding, LSTM, Dense
)

from tensorflow.keras.callbacks import EarlyStopping

# STEP 1: Load text dataset
data = pd.read_csv("sentiment_dataset.csv")

texts = (
    data["Review"]
    .dropna()
    .astype(str)
    .tolist()
)

if not texts:
    raise ValueError("No review text found in the dataset.")

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

print("Total sentences:", len(texts))

# STEP 2: Tokenization and vocabulary generation
MAX_WORDS = 5000
MAX_TEXT_LENGTH = 50

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_TEXT_LENGTH
)

# Learn the vocabulary
vectorizer.adapt(texts)

vocabulary = vectorizer.get_vocabulary()

total_words = len(vocabulary)

print("\nVocabulary size:", total_words)

# STEP 3: Generate word sequences
encoded_sentences = vectorizer(
    tf.constant(texts)
).numpy()

input_sequences = []

for sentence in encoded_sentences:

    # Remove padding tokens
    token_list = [
        int(token) for token in sentence if token != 0
    ]

    # Create progressive sequences
    for i in range(1, len(token_list)):

        ngram_sequence = token_list[:i + 1]

        input_sequences.append(ngram_sequence)

print("\nTotal training sequences:", len(input_sequences))

if not input_sequences:
    raise ValueError("Not enough words to create training sequences.")

# STEP 4: Padding
max_sequence_len = max(
    len(sequence) for sequence in input_sequences
)

padded_sequences = tf.keras.utils.pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

print("Maximum sequence length:", max_sequence_len)

# STEP 5: Separate input and target
X = padded_sequences[:, :-1]

y = padded_sequences[:, -1]

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)

# STEP 6: Build LSTM architecture
model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(max_sequence_len - 1,),
        dtype="int32"
    ),

    # Embedding layer
    Embedding(
        input_dim=total_words,
        output_dim=64,
        mask_zero=True
    ),

    # LSTM layer
    LSTM(
        units=100
    ),

    # Predict the next word
    Dense(
        total_words,
        activation="softmax"
    )
])

# STEP 7: Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nMODEL ARCHITECTURE")
model.summary()

# STEP 8: Train the model
early_stop = EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)

print("\nTraining LSTM model...")

history = model.fit(
    X,
    y,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining completed!")

# STEP 9: Next word prediction function
index_to_word = {
    index: word
    for index, word in enumerate(vocabulary)
}

def generate_text(seed_text, next_words=5):

    for _ in range(next_words):

        # Convert input text into word IDs
        token_list = vectorizer(
            tf.constant([seed_text])
        ).numpy()[0]

        token_list = [
            int(token) for token in token_list
            if token != 0
        ]

        # Pad the input sequence
        token_list = tf.keras.utils.pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre",
            truncating="pre"
        )

        # Predict next word probabilities
        predicted_probs = model.predict(
            token_list,
            verbose=0
        )[0]

        # Avoid predicting padding or unknown tokens
        predicted_probs[0] = 0.0

        if total_words > 1:
            predicted_probs[1] = 0.0

        predicted_index = int(
            np.argmax(predicted_probs)
        )

        # Convert predicted ID into a word
        predicted_word = index_to_word.get(
            predicted_index, ""
        )

        if not predicted_word:
            break

        # Append predicted word
        seed_text += " " + predicted_word

    return seed_text

# STEP 10: Generate text
print("\nNEXT WORD PREDICTION")

seed_text = "the product"

print("Input text:", seed_text)

generated_text = generate_text(
    seed_text,
    next_words=5
)

print("\nGenerated text:")
print(generated_text)

# STEP 11: Plot training loss
plt.figure(figsize=(7, 4))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.title("LSTM Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 12: Save model
model.save("next_word_lstm.keras")

print("\nModel saved successfully!")
