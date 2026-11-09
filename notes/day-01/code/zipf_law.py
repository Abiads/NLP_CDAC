import nltk
nltk.download('gutenberg')
nltk.download('punkt')
import collections
import matplotlib.pyplot as plt

nltk.download('gutenberg')

words = [
    w.lower()
    for w in nltk.corpus.gutenberg.words("austen-emma.txt")
]

freq = collections.Counter(words).most_common()

ranks = range(1, len(freq) + 1)

plt.loglog(
    ranks,
    [f for _, f in freq]
)

plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf's Law for Emma")
plt.show()

print(freq[:10])