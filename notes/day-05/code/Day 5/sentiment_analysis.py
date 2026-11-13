
# SENTIMENT ANALYSIS USING LOGISTIC REGRESSION AND SVM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load dataset
data = pd.read_csv("sentiment_dataset.csv")

X = data["Review"]
y = data["Sentiment"]

# STEP 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Convert text into numerical features
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# STEP 4: Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# STEP 5: Train SVM
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

# STEP 6: Make predictions
lr_pred = lr_model.predict(X_test_tfidf)
svm_pred = svm_model.predict(X_test_tfidf)

# STEP 7: Evaluate models
print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\nSVM RESULTS")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# STEP 8: Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.heatmap(
    confusion_matrix(y_test, lr_pred),
    annot=True, fmt="d", cmap="Blues", ax=axes[0]
)
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    confusion_matrix(y_test, svm_pred),
    annot=True, fmt="d", cmap="Greens", ax=axes[1]
)
axes[1].set_title("SVM")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# STEP 9: Predict sentiment of new reviews
new_reviews = [
    "This product is excellent and amazing",
    "The product is very poor and disappointing"
]

new_tfidf = tfidf.transform(new_reviews)

print("\nNEW REVIEW PREDICTIONS")

print("Logistic Regression:",
      lr_model.predict(new_tfidf))

print("SVM:",
      svm_model.predict(new_tfidf))
