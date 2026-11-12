
import nltk
import networkx as nx
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Sample document
text = """
Artificial Intelligence is transforming the modern world.
It is widely used in healthcare, education, agriculture, and industry.
Machine learning is an important branch of artificial intelligence.
It enables computers to learn from data and make predictions.
Deep learning uses neural networks to solve complex problems.
Artificial intelligence helps doctors analyze medical images.
It also supports personalized learning in education.
AI technologies are improving productivity and decision making.
However, artificial intelligence also raises ethical concerns.
Researchers are developing responsible and trustworthy AI systems.
"""

# Step 1: Split text into sentences
sentences = sent_tokenize(text)

# Step 2: Preprocess sentences
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(sentence):
    words = word_tokenize(sentence.lower())

    words = [
        stemmer.stem(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return words

processed_sentences = [
    preprocess(sentence) for sentence in sentences
]

# Step 3: Calculate sentence similarity
def sentence_similarity(sentence1, sentence2):

    words1 = set(sentence1)
    words2 = set(sentence2)

    if not words1 or not words2:
        return 0

    common_words = words1.intersection(words2)

    similarity = len(common_words) / (
        len(words1) + len(words2)
    )

    return similarity

# Step 4: Create sentence similarity graph
graph = nx.Graph()

for i in range(len(sentences)):
    graph.add_node(i)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):

        similarity = sentence_similarity(
            processed_sentences[i],
            processed_sentences[j]
        )

        if similarity > 0:
            graph.add_edge(i, j, weight=similarity)

# Step 5: Apply TextRank using PageRank
scores = nx.pagerank(
    graph,
    weight="weight"
)

# Step 6: Rank sentences
ranked_sentences = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True
)

# Step 7: Select top sentences
summary_length = 3

selected_indices = [
    index for index, score in ranked_sentences[:summary_length]
]

# Preserve original document order
selected_indices.sort()

summary = " ".join(
    sentences[index] for index in selected_indices
)

# Display results
print("\nAUTOMATIC TEXT SUMMARIZATION")
print("--------------------------------")

print("\nOriginal Document:")
print(text)

print("\nSentence Scores:")
for index, score in ranked_sentences:
    print(
        f"Sentence {index + 1}: "
        f"{score:.4f} - {sentences[index]}"
    )

print("\nGenerated Summary:")
print(summary)
