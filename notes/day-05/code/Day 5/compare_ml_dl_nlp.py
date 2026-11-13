
# COMPARATIVE ANALYSIS OF ML AND DL MODELS FOR NLP

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding, SimpleRNN, LSTM,
    TextVectorization
)

from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# STEP 1: Load dataset
data = pd.read_csv("sentiment_dataset.csv")

data = data.dropna(subset=["Review", "Sentiment"])

X = data["Review"].astype(str).to_numpy()

y = (
    data["Sentiment"]
    .str.strip()
    .str.lower()
    .map({"negative": 0, "positive": 1})
)

if y.isna().any():
    raise ValueError("Invalid sentiment labels found.")

y = y.to_numpy(dtype="int32")

print("Total reviews:", len(X))

# STEP 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Store final results
results = []
confusion_matrices = {}
training_histories = {}

# Common evaluation function
def evaluate_model(name, actual, predicted):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted,
        zero_division=0
    )

    recall = recall_score(
        actual, predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual, predicted,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    confusion_matrices[name] = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    print("\n", name, "RESULTS")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))


# STEP 3: TF-IDF feature extraction
# Keep negation words such as "not" and "no".

tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# STEP 4: NAIVE BAYES
print("\nTRAINING NAIVE BAYES")

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)

evaluate_model("Naive Bayes", y_test, nb_pred)


# STEP 5: SUPPORT VECTOR MACHINE
print("\nTRAINING SVM")

svm_model = LinearSVC(
    random_state=42
)

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

evaluate_model("SVM", y_test, svm_pred)


# STEP 6: ARTIFICIAL NEURAL NETWORK (ANN)
print("\nTRAINING ANN")

# Convert sparse TF-IDF into dense numerical arrays
X_train_dense = X_train_tfidf.toarray().astype("float32")
X_test_dense = X_test_tfidf.toarray().astype("float32")

ann_model = tf.keras.Sequential([
    Input(shape=(X_train_dense.shape[1],)),

    Dense(64, activation="relu"),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

ann_history = ann_model.fit(
    X_train_dense,
    y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

training_histories["ANN"] = ann_history

ann_prob = ann_model.predict(
    X_test_dense,
    verbose=0
).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)

evaluate_model("ANN", y_test, ann_pred)


# STEP 7: PREPARE WORD SEQUENCES FOR RNN AND LSTM

MAX_WORDS = 3000
MAX_LENGTH = 40

train_text = tf.constant(X_train, dtype=tf.string)
test_text = tf.constant(X_test, dtype=tf.string)

def create_sequence_vectorizer():

    vectorizer = TextVectorization(
        max_tokens=MAX_WORDS,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=MAX_LENGTH
    )

    vectorizer.adapt(train_text)

    return vectorizer


# Common function to build RNN or LSTM
def build_sequence_model(model_type):

    vectorizer = create_sequence_vectorizer()

    vocab_size = len(
        vectorizer.get_vocabulary()
    )

    layers = [
        Input(shape=(), dtype=tf.string),
        vectorizer,
        Embedding(
            input_dim=vocab_size,
            output_dim=64,
            mask_zero=True
        )
    ]

    if model_type == "RNN":

        layers.append(
            SimpleRNN(
                32,
                activation="tanh"
            )
        )

    elif model_type == "LSTM":

        layers.append(
            LSTM(
                32,
                activation="tanh"
            )
        )

    layers.append(
        Dense(1, activation="sigmoid")
    )

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# STEP 8: RECURRENT NEURAL NETWORK
print("\nTRAINING RNN")

rnn_model = build_sequence_model("RNN")

rnn_history = rnn_model.fit(
    train_text,
    y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

training_histories["RNN"] = rnn_history

rnn_prob = rnn_model.predict(
    test_text,
    verbose=0
).ravel()

rnn_pred = (rnn_prob >= 0.5).astype(int)

evaluate_model("RNN", y_test, rnn_pred)


# STEP 9: LONG SHORT-TERM MEMORY
print("\nTRAINING LSTM")

lstm_model = build_sequence_model("LSTM")

lstm_history = lstm_model.fit(
    train_text,
    y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

training_histories["LSTM"] = lstm_history

lstm_prob = lstm_model.predict(
    test_text,
    verbose=0
).ravel()

lstm_pred = (lstm_prob >= 0.5).astype(int)

evaluate_model("LSTM", y_test, lstm_pred)


# STEP 10: FINAL PERFORMANCE COMPARISON
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")

print(results_df.round(4).to_string(index=False))


# STEP 11: PERFORMANCE COMPARISON GRAPH
plt.figure(figsize=(11, 6))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(11, 6))

plt.title("ML and DL Models - Performance Comparison")
plt.ylabel("Performance Score")
plt.xlabel("Models")
plt.ylim(0, 1.1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# STEP 12: CONFUSION MATRICES
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (name, cm) in zip(
    axes.flat, confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

axes.flat[-1].axis("off")

plt.tight_layout()
plt.show()


# STEP 13: TRAINING LOSS COMPARISON
plt.figure(figsize=(9, 5))

for name, history in training_histories.items():

    plt.plot(
        history.history["loss"],
        label=name
    )

plt.title("Deep Learning Models - Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# STEP 14: SAVE RESULTS
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nComparison results saved successfully!")
