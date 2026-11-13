
# COMPARATIVE ANALYSIS OF TEXT CLASSIFICATION ALGORITHMS
# Algorithms: Naive Bayes, SVM and Centroid-Based Classification

# STEP 1: Import libraries

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# STEP 2: Create a common text dataset

documents = [

    # TECHNOLOGY - 10 DOCUMENTS
    "Artificial intelligence is transforming healthcare",
    "Machine learning is used in data science",
    "Python programming is useful for software development",
    "Deep learning is a branch of artificial intelligence",
    "Computer networks connect different devices",
    "Cloud computing provides online services",
    "Cybersecurity protects computer systems",
    "Robotics uses artificial intelligence",
    "Data mining discovers useful information",
    "Software engineering develops computer applications",

    # SPORTS - 10 DOCUMENTS
    "India won the cricket match",
    "Football is a popular sport",
    "The cricket team scored a century",
    "The football player scored a goal",
    "The team won the basketball tournament",
    "The tennis player won the championship",
    "The coach trained the cricket players",
    "The goalkeeper saved the penalty",
    "Athletes participated in the sports competition",
    "The player scored points in the match",

    # EDUCATION - 10 DOCUMENTS
    "Students attend classes in college",
    "Teachers explain programming concepts",
    "University conducts semester examinations",
    "Students prepare assignments and projects",
    "Online education provides learning opportunities",
    "The professor teaches computer science",
    "Students submit their laboratory records",
    "The college organizes technical workshops",
    "Teachers conduct practical examinations",
    "Students attend lectures in the classroom"
]


# STEP 3: Define category labels

labels = (
    ["Technology"] * 10 +
    ["Sports"] * 10 +
    ["Education"] * 10
)


# STEP 4: Divide data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)


print("Training documents:", len(X_train))
print("Testing documents:", len(X_test))


# STEP 5: Convert text into TF-IDF features

vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# Store actual labels as an array

y_train = np.array(y_train)
y_test = np.array(y_test)


# STEP 6: Create the three classification models

# MODEL 1: NAIVE BAYES

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)


# MODEL 2: SUPPORT VECTOR MACHINE

svm_model = SVC(kernel="linear")

svm_model.fit(X_train_tfidf, y_train)

svm_predictions = svm_model.predict(X_test_tfidf)


# MODEL 3: CENTROID-BASED CLASSIFICATION

categories = np.unique(y_train)

centroids = []

for category in categories:

    category_documents = X_train_tfidf[
        y_train == category
    ]

    centroid = category_documents.mean(axis=0)

    centroids.append(np.asarray(centroid).ravel())


centroids = np.array(centroids)


# Calculate similarity between test documents and centroids

similarity = cosine_similarity(
    X_test_tfidf,
    centroids
)


# Select category with highest similarity

centroid_predictions = categories[
    np.argmax(similarity, axis=1)
]


# STEP 7: Create a common evaluation function

def evaluate_model(actual, predicted):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    return accuracy, precision, recall, f1


# STEP 8: Evaluate all three algorithms

nb_results = evaluate_model(y_test, nb_predictions)

svm_results = evaluate_model(y_test, svm_predictions)

centroid_results = evaluate_model(
    y_test,
    centroid_predictions
)


# STEP 9: Display comparative performance

comparison = pd.DataFrame({

    "Algorithm": [
        "Naive Bayes",
        "SVM",
        "Centroid-Based"
    ],

    "Accuracy": [
        nb_results[0],
        svm_results[0],
        centroid_results[0]
    ],

    "Precision": [
        nb_results[1],
        svm_results[1],
        centroid_results[1]
    ],

    "Recall": [
        nb_results[2],
        svm_results[2],
        centroid_results[2]
    ],

    "F1-Score": [
        nb_results[3],
        svm_results[3],
        centroid_results[3]
    ]

})


print("\n===== COMPARATIVE PERFORMANCE =====")

print(comparison.to_string(index=False))


# STEP 10: Display classification reports

models = {
    "Naive Bayes": nb_predictions,
    "SVM": svm_predictions,
    "Centroid-Based": centroid_predictions
}


for name, predictions in models.items():

    print("\n===== ", name, " =====")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))


# STEP 11: Display confusion matrices

for name, predictions in models.items():

    print("\nCONFUSION MATRIX:", name)

    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Education", "Sports", "Technology"]
    ))


# STEP 12: Identify the best-performing algorithm

best_index = comparison["F1-Score"].idxmax()

best_algorithm = comparison.loc[
    best_index, "Algorithm"
]

print("\n===== BEST PERFORMING ALGORITHM =====")

print("Based on weighted F1-score:", best_algorithm)


# STEP 13: Classify new documents using all three algorithms

new_documents = [

    "Artificial intelligence helps computers learn",

    "India won the football tournament",

    "Students attend university classes"

]


new_tfidf = vectorizer.transform(new_documents)


nb_new = nb_model.predict(new_tfidf)

svm_new = svm_model.predict(new_tfidf)


centroid_similarity = cosine_similarity(
    new_tfidf,
    centroids
)

centroid_new = categories[
    np.argmax(centroid_similarity, axis=1)
]


print("\n===== NEW DOCUMENT PREDICTIONS =====")


for i, document in enumerate(new_documents):

    print("\nDocument:", document)

    print("Naive Bayes:", nb_new[i])

    print("SVM:", svm_new[i])

    print("Centroid-Based:", centroid_new[i])
