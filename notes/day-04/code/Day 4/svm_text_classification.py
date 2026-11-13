from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Training Data
documents = [
    "machine learning improves prediction",
    "deep learning uses neural networks",
    "artificial intelligence is powerful",
    "football is a popular sport",
    "cricket is played worldwide",
    "soccer is an exciting game"
]

# Labels
labels = [
    "Technology",
    "Technology",
    "Technology",
    "Sports",
    "Sports",
    "Sports"
]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

# SVM Classifier
model = SVC(kernel='linear')

model.fit(X, labels)

# Test Data
test_docs = [
    "neural network and machine learning",
    "football world cup"
]

actual_labels = [
    "Technology",
    "Sports"
]

# Transform Test Data
X_test = vectorizer.transform(test_docs)

# Prediction
predicted_labels = model.predict(X_test)

print("Predictions:")
for doc, pred in zip(test_docs, predicted_labels):
    print(doc, "->", pred)

# Evaluation
accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    average='weighted'
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    average='weighted'
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    average='weighted'
)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)