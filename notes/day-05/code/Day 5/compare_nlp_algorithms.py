# COMPARISON OF MACHINE LEARNING ALGORITHMS FOR NLP TASKS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# STEP 1: Load Dataset (or generate a fallback sample dataset if missing)
csv_filename = "sentiment_dataset.csv"

if os.path.exists(csv_filename):
    data = pd.read_csv(csv_filename)
else:
    print(f"Warning: '{csv_filename}' not found. Generating a sample dataset to demonstrate code functionality...")
    # Creating dummy movie/product reviews
    sample_data = {
        "Review": [
            "This movie was absolutely amazing and I loved it!",
            "Superb acting and a brilliant storyline. Highly recommend.",
            "Wonderful experience, easily one of the best films of the year.",
            "An excellent production with great cast and visuals.",
            "Highly entertaining, I will definitely watch this again!",
            "Worst movie I have ever seen. Total waste of time.",
            "Terrible screenplay, bad acting, and completely boring.",
            "I hated this film. It made absolutely no sense.",
            "A frustrating waste of money with an awful ending.",
            "Very disappointing. The plot was weak and uninspiring."
        ] * 10,  # Duplicate to create a larger sample dataset
        "Sentiment": [
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Negative", "Negative", "Negative", "Negative", "Negative"
        ] * 10
    }
    data = pd.DataFrame(sample_data)

X = data["Review"]
y = data["Sentiment"]

# STEP 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Define Machine Learning Algorithms
models = {
    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    )
}

results = []

# STEP 4: Train and Evaluate Each Model
for name, model in models.items():

    print("\nTraining:", name)

    # TF-IDF + Machine Learning Model
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )),
        ("classifier", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict test data
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    # Display confusion matrix
    cm = confusion_matrix(
        y_test, predictions,
        labels=["Negative", "Positive"]
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# STEP 5: Display Final Comparison
results_df = pd.DataFrame(results)

print("\nFINAL ALGORITHM COMPARISON")
print(results_df.to_string(index=False))

# STEP 6: Plot Performance Comparison
results_df.set_index("Algorithm")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Comparison of Machine Learning Algorithms")
plt.ylabel("Performance Score")
plt.xlabel("Algorithms")
plt.ylim(0, 1.1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()