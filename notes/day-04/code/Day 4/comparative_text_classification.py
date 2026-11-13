from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------------------------------
# Dataset
# --------------------------------------------------

documents = [
    "machine learning and artificial intelligence",
    "deep learning neural networks",
    "natural language processing",
    "data science and machine learning",
    "artificial intelligence systems",
    "neural network applications",

    "football world cup tournament",
    "cricket match highlights",
    "soccer championship players",
    "basketball league tournament",
    "tennis grand slam match",
    "football coaching techniques"
]

labels = [
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",
    "Technology",

    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports"
]

# --------------------------------------------------
# TF-IDF Feature Extraction
# --------------------------------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# --------------------------------------------------
# Naive Bayes
# --------------------------------------------------

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

# --------------------------------------------------
# SVM
# --------------------------------------------------

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

# --------------------------------------------------
# Centroid Classifier
# --------------------------------------------------

categories = list(set(y_train))
centroids = {}

for category in categories:

    indices = [
        i for i in range(len(y_train))
        if y_train[i] == category
    ]

    centroid = np.asarray(
        X_train[indices].mean(axis=0)
    )

    centroids[category] = centroid

centroid_predictions = []

for doc in X_test:

    best_category = None
    best_similarity = -1

    for category in categories:

        similarity = cosine_similarity(
            doc,
            centroids[category]
        )[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_category = category

    centroid_predictions.append(best_category)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate(name, actual, predicted):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Predictions :", predicted)

    print("Accuracy  :",
          round(accuracy_score(actual, predicted), 4))

    print("Precision :",
          round(
              precision_score(
                  actual,
                  predicted,
                  average='weighted',
                  zero_division=0
              ), 4
          ))

    print("Recall    :",
          round(
              recall_score(
                  actual,
                  predicted,
                  average='weighted',
                  zero_division=0
              ), 4
          ))

    print("F1 Score  :",
          round(
              f1_score(
                  actual,
                  predicted,
                  average='weighted',
                  zero_division=0
              ), 4
          ))

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nActual Test Labels:")
print(y_test)

evaluate(
    "Naive Bayes",
    y_test,
    nb_predictions
)

evaluate(
    "Support Vector Machine",
    y_test,
    svm_predictions
)

evaluate(
    "Centroid Classifier",
    y_test,
    centroid_predictions
)