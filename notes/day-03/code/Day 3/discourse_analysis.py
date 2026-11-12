import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load pretrained sentence embedding model
print("Loading sentence similarity model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text
text = """
Ravi studied hard for his examination.
Therefore, he scored excellent marks.
However, he found mathematics difficult.
Moreover, he practiced additional problems.
As a result, he improved his performance.
"""

# Identify discourse markers
discourse_markers = [
    "however",
    "therefore",
    "because",
    "moreover",
    "although",
    "furthermore",
    "as a result",
    "in addition",
    "consequently",
    "but"
]

# Split text into sentences
sentences = sent_tokenize(text)

print("\nDISCOURSE ANALYSIS AND COHERENCE")
print("--------------------------------")

print("\nSentences:")
for i, sentence in enumerate(sentences, start=1):
    print(f"{i}. {sentence}")

# Identify discourse markers in each sentence
print("\nDiscourse Marker Analysis:")

for i, sentence in enumerate(sentences, start=1):

    lower_sentence = sentence.lower()

    found_markers = [
        marker for marker in discourse_markers
        if marker in lower_sentence
    ]

    if found_markers:
        print(f"Sentence {i}: {found_markers}")
    else:
        print(f"Sentence {i}: No explicit marker found")

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Calculate pairwise sentence similarity
similarity_matrix = cosine_similarity(embeddings)

print("\nSentence Similarity Matrix:")
print(np.round(similarity_matrix, 2))

# Analyze consecutive sentence relationships
print("\nConsecutive Sentence Coherence:")

coherence_scores = []

for i in range(len(sentences) - 1):

    score = similarity_matrix[i][i + 1]

    coherence_scores.append(score)

    print(f"\nSentence {i + 1}: {sentences[i]}")
    print(f"Sentence {i + 2}: {sentences[i + 1]}")
    print(f"Similarity Score: {score:.4f}")

# Calculate average consecutive similarity
if coherence_scores:

    average_coherence = np.mean(coherence_scores)

    print("\nAverage Consecutive Sentence Similarity:")
    print(f"{average_coherence:.4f}")

print("\nDiscourse Analysis Completed.")