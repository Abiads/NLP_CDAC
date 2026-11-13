
# TEXT CLUSTERING USING K-MEANS

# STEP 1: Import required libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import pandas as pd
from collections import Counter


# STEP 2: Create sample text documents

documents = [

    # Technology-related documents
    "Artificial intelligence is transforming healthcare",
    "Machine learning is used in data science",
    "Python programming is useful for software development",
    "Deep learning is a branch of artificial intelligence",
    "Computer networks connect different devices",

    # Sports-related documents
    "India won the cricket match",
    "Football is a popular sport",
    "The cricket team scored a century",
    "The football player scored a goal",
    "The team won the basketball tournament",

    # Education-related documents
    "Students attend classes in college",
    "Teachers explain programming concepts",
    "University conducts semester examinations",
    "Students prepare assignments and projects",
    "Online education provides learning opportunities"
]


# STEP 3: Convert text into TF-IDF features

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(documents)


print("TF-IDF FEATURE MATRIX:")

print(X.toarray())


# STEP 4: Apply K-Means clustering

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X)


# STEP 5: Get cluster assignments

cluster_labels = kmeans.labels_


# STEP 6: Display documents and their clusters

print("\n===== DOCUMENT CLUSTERING RESULTS =====")

results = pd.DataFrame({

    "Document": documents,

    "Cluster": cluster_labels

})

print(results.to_string(index=False))


# STEP 7: Count documents in each cluster

print("\n===== NUMBER OF DOCUMENTS IN EACH CLUSTER =====")

cluster_counts = Counter(cluster_labels)

for cluster, count in sorted(cluster_counts.items()):

    print("Cluster", cluster, ":", count, "documents")


# STEP 8: Display important words in each cluster

print("\n===== IMPORTANT WORDS IN EACH CLUSTER =====")

terms = vectorizer.get_feature_names_out()

for i, center in enumerate(kmeans.cluster_centers_):

    top_indices = center.argsort()[-5:][::-1]

    top_words = [terms[index] for index in top_indices]

    print("\nCluster", i)

    print("Important words:", top_words)


# STEP 9: Predict cluster for a new document

new_documents = [

    "Artificial intelligence and machine learning",

    "India won the football tournament",

    "Students attend university classes"

]

new_features = vectorizer.transform(new_documents)

predicted_clusters = kmeans.predict(new_features)


print("\n===== NEW DOCUMENT CLUSTERING =====")

for document, cluster in zip(
    new_documents, predicted_clusters
):

    print("\nDocument:", document)

    print("Assigned Cluster:", cluster)
