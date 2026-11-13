
# IMPLEMENTATION OF LSTM FOR SENTIMENT ANALYSIS

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

X = data["Review"].fillna("").astype(str).to_numpy()

# Convert Positive = 1 and Negative = 0
y = (
    data["Sentiment"]
    .str.strip()
    .str.lower()
    .map({"negative": 0, "positive": 1})
)

if y.isna().any():
    raise ValueError("Invalid sentiment labels found.")

y = y.to_numpy(dtype="float32")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# STEP 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Text preprocessing
MAX_WORDS = 3000
MAX_LENGTH = 40

# Convert text into TensorFlow string tensors
train_text = tf.constant(X_train, dtype=tf.string)
test_text = tf.constant(X_test, dtype=tf.string)

# Create vocabulary and token sequences
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

# Learn vocabulary using training data only
vectorizer.adapt(train_text)

vocab_size = len(vectorizer.get_vocabulary())

print("Vocabulary Size:", vocab_size)

# STEP 4: Build LSTM architecture
model = tf.keras.Sequential([

    tf.keras.Input(shape=(), dtype=tf.string),

    vectorizer,

    # Embedding layer
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),

    # LSTM layer
    tf.keras.layers.LSTM(
        units=64,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=0.2,
        return_sequences=False
    ),

    # Output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# STEP 5: Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nLSTM MODEL ARCHITECTURE")
model.summary()

# STEP 6: Train model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining LSTM model...")

history = model.fit(
    train_text,
    y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining completed!")

# STEP 7: Evaluate model
test_loss, test_accuracy = model.evaluate(
    test_text,
    y_test,
    verbose=0
)

print("\nLSTM MODEL PERFORMANCE")

print("Test Loss:", round(test_loss, 4))
print("Accuracy:", round(test_accuracy, 4))

# STEP 8: Make predictions
probabilities = model.predict(test_text).ravel()

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

plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# STEP 10: Plot training loss
plt.figure(figsize=(7, 4))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# STEP 11: Predict new reviews
new_reviews = tf.constant([
    "This product is excellent and amazing",
    "The product is very poor and disappointing"
], dtype=tf.string)

new_probabilities = model.predict(new_reviews).ravel()

print("\nNEW REVIEW PREDICTIONS")

for review, probability in zip(
    new_reviews.numpy(), new_probabilities
):

    review = review.decode("utf-8")

    sentiment = (
        "Positive" if probability >= 0.5
        else "Negative"
    )

    print("\nReview:", review)
    print("Positive Probability:", round(float(probability), 4))
    print("Predicted Sentiment:", sentiment)

# STEP 12: Save the trained model
model.save("lstm_sentiment_model.keras")

print("\nLSTM model saved successfully!")
