from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Training data
documents = [
    "machine learning improves prediction",
    "deep learning uses neural networks",
    "football is a popular sport",
    "cricket is played worldwide",
    "artificial intelligence is powerful",
    "soccer is an exciting game"
]

# Categories
labels = [
    "Technology",
    "Technology",
    "Sports",
    "Sports",
    "Technology",
    "Sports"
]

# TF-IDF + Naive Bayes Model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(documents, labels)

# Test document
test_text = ["neural network and artificial intelligence"]

prediction = model.predict(test_text)

print("Predicted Category:", prediction[0])