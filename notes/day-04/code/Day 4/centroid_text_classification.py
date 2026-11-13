from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Training documents
documents = [
    "machine learning and artificial intelligence",
    "deep learning neural networks",
    "natural language processing",
    "football world cup tournament",
    "cricket match highlights",
    "soccer championship players"
]

# Categories
labels = [
    "Technology",
    "Technology",
    "Technology",
    "Sports",
    "Sports",
    "Sports"
]

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

# Calculate category centroids
categories = list(set(labels))
centroids = {}

for category in categories:
    indices = [i for i in range(len(labels))
               if labels[i] == category]

    centroid = X[indices].mean(axis=0)
    centroids[category] = centroid

# Test document
test_doc = ["neural network and machine learning"]

# Convert test document
test_vector = vectorizer.transform(test_doc)

# Compute similarities
best_category = None
highest_similarity = -1

for category in categories:
    similarity = cosine_similarity(
        test_vector,
        centroids[category]
    )[0][0]

    print(category,
          "Similarity:",
          round(similarity, 4))

    if similarity > highest_similarity:
        highest_similarity = similarity
        best_category = category

print("\nPredicted Category:",
      best_category)
