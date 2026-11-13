
# IMPLEMENTATION OF RNN (LSTM) FOR TEXT SEQUENCE PROCESSING

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load dataset
data = pd.read_csv("sentiment_dataset.csv")

X = data["Review"].astype(str).to_numpy()

# Convert Positive = 1 and Negative = 0
y = (
    data["Sentiment"]
    .str.strip()
    .str.lower()
    .map({"negative": 0, "positive": 1})
)

if y.isna().any():
    raise ValueError("Dataset contains invalid sentiment labels.")

y = y.to_numpy(dtype="int32")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# STEP 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Convert sentences into word sequences
MAX_WORDS = 3000
MAX_LENGTH = 30

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

# Learn vocabulary only from training data
vectorizer.adapt(X_train)

vocab_size = len(vectorizer.get_vocabulary())

print("Vocabulary size:", vocab_size)

# STEP 4: Build LSTM architecture (handles masking natively and avoids RNN limitations)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),

    vectorizer,

    # Embedding layer with mask_zero enabled
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        mask_zero=True
    ),

    # LSTM layer supports masking perfectly and works better for text sequential modeling
    tf.keras.layers.LSTM(
        units=32,
        activation="tanh",
        return_sequences=False
    ),

    # Output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# STEP 5: Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nLSTM MODEL ARCHITECTURE")
model.summary()

# STEP 6: Train the model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining model...")

history = model.fit(
    X_train.reshape(-1, 1),
    y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining completed!")

# STEP 7: Evaluate the model
test_loss, test_accuracy = model.evaluate(
    X_test.reshape(-1, 1),
    y_test,
    verbose=0
)

print("\nMODEL PERFORMANCE")
print("Test Loss:", round(test_loss, 4))
print("Accuracy:", round(test_accuracy, 4))

# STEP 8: Predict test data
probabilities = model.predict(
    X_test.reshape(-1, 1)
).ravel()

predictions = (probabilities >= 0.5).astype(int)

print("Precision:", precision_score(
    y_test, predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, predictions, zero_division=0
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

# STEP 9: Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# STEP 10: Training and validation loss
plt.figure(figsize=(7, 4))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 11: Predict new reviews
new_reviews = [
    "This product is excellent and amazing",
    "The product is very poor and disappointing"
]

new_input = np.array(new_reviews).reshape(-1, 1)

new_probabilities = model.predict(new_input).ravel()

print("\nNEW REVIEW PREDICTIONS")

for review, probability in zip(
    new_reviews, new_probabilities
):
    sentiment = (
        "Positive" if probability >= 0.5
        else "Negative"
    )

    print("\nReview:", review)
    print("Positive Probability:", round(float(probability), 4))
    print("Predicted Sentiment:", sentiment)

# STEP 12: Save trained model
model.save("lstm_sentiment_model.keras")

print("\nModel saved successfully!")