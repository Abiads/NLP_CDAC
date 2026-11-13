from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample documents
documents = [
    "machine learning and artificial intelligence",
    "deep learning using neural networks",
    "natural language processing techniques",
    "football world cup tournament",
    "cricket match and batting skills",
    "soccer players and championship"
]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

# Create K-Means model
k = 2

model = KMeans(
    n_clusters=k,
    random_state=42
)

# Train model
model.fit(X)

# Display clusters
for i in range(len(documents)):
    print("Document:", documents[i])
    print("Cluster :", model.labels_[i])
    print()