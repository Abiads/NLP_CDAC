# IMPLEMENTATION OF NEURAL NETWORK FOR TEXT CLASSIFICATION

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Programmatically Create and Load Dataset in-place
mock_data = {
    "Review": [
        "This product is absolutely amazing and works perfectly!",
        "I highly recommend this to everyone, excellent quality.",
        "Super fast shipping and great customer service.",
        "Best purchase I have made this year, extremely satisfied.",
        "This is a wonderful product, 10/10.",
        "This product is very poor and highly disappointing.",
        "Worst purchase ever, it broke on the first day.",
        "Terrible quality and the customer support was useless.",
        "I waste my money on this product, do not buy it.",
        "Completely useless and does not work as advertised."
    ],
    "Sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]
}

data = pd.DataFrame(mock_data)

X = data["Review"].astype(str)
y = data["Sentiment"].astype(str).str.strip().str.title()

# STEP 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Create ANN Model
ann_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=3000
    )),

    ("ann", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    ))
])

# STEP 4: Train the Neural Network
print("Training ANN model...")

ann_pipeline.fit(X_train, y_train)

print("Training completed!")

# STEP 5: Predict Test Data
y_pred = ann_pipeline.predict(X_test)

# STEP 6: Evaluate Model
print("\nANN MODEL PERFORMANCE")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(
          y_test, y_pred,
          average="weighted",
          zero_division=0
      ))

print("Recall:",
      recall_score(
          y_test, y_pred,
          average="weighted",
          zero_division=0
      ))

print("F1 Score:",
      f1_score(
          y_test, y_pred,
          average="weighted",
          zero_division=0
      ))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# STEP 7: Display Confusion Matrix
cm = confusion_matrix(
    y_test, y_pred,
    labels=["Negative", "Positive"]
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

plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# STEP 8: Visualize Training Loss
ann_model = ann_pipeline.named_steps["ann"]

plt.figure(figsize=(7, 4))

plt.plot(
    ann_model.loss_curve_,
    label="Training Loss"
)

plt.title("ANN Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 9: Predict New Reviews
new_reviews = [
    "This product is excellent and amazing",
    "The product is very poor and disappointing"
]

predictions = ann_pipeline.predict(new_reviews)
probabilities = ann_pipeline.predict_proba(new_reviews)

print("\nNEW REVIEW PREDICTIONS")

for review, prediction, probability in zip(
    new_reviews, predictions, probabilities
):
    print("\nReview:", review)
    print("Predicted Sentiment:", prediction)

    for label, score in zip(
        ann_model.classes_, probability
    ):
        print(label, "Probability:", round(score, 4))