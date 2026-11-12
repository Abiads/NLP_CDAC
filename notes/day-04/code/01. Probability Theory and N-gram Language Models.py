"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 4 / Session 5
Topic:  Experiment 1: Probability Theory and N-gram Language Models
File:   01. Probability Theory and N-gram Language Models.py
=============================================================================

WHAT IS AN N-GRAM LANGUAGE MODEL?
---------------------------------
An N-gram language model assigns a probability P(W) to a sequence of words
W = (w1, w2, ..., wm) to quantify how natural, fluent, or likely the sentence
is in a given natural language.

PROBABILITY CHAIN RULE:
-----------------------
By the chain rule of probability:
  P(w1, w2, ..., wm) = P(w1) * P(w2|w1) * P(w3|w1, w2) ... * P(wm|w1...wm-1)

MARKOV ASSUMPTIONS:
-------------------
Computing full history probabilities becomes intractable due to data sparsity.
The Markov assumption limits history to the previous (N - 1) words:
  1. Unigram Model (N=1, No memory / Independent):
       P(w1...wm) = Product_{i=1}^m P(wi) = Product_{i=1}^m ( Count(wi) / Total_Words )
  2. Bigram Model (N=2, First-order Markov / 1-word memory):
       P(w1...wm) = Product_{i=1}^m P(wi | wi-1) = Product_{i=1}^m ( Count(wi-1, wi) / Count(wi-1) )
  3. Trigram Model (N=3, Second-order Markov / 2-word memory):
       P(w1...wm) = Product_{i=1}^m P(wi | wi-2, wi-1) = Product_{i=1}^m ( Count(wi-2, wi-1, wi) / Count(wi-2, wi-1) )

SENTENCE BOUNDARY PADDING:
--------------------------
To ensure mathematically valid probability distributions over variable-length
sentences, we pad sentences with start tokens '<s>' and end tokens '</s>'.
=============================================================================
"""

from collections import Counter
from typing import List, Tuple, Dict


class NGramLanguageModel:
    """
    A foundational N-gram Language Model supporting Unigram, Bigram, and Trigram
    maximum likelihood estimation with consistent sentence boundary padding.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.total_words = 0
        self.num_sentences = len(corpus)
        self._train()

    def _train(self):
        """Tokenize corpus and accumulate padded n-gram frequencies."""
        for sentence in self.corpus:
            tokens = sentence.lower().split()
            # Padded tokens for bigram and trigram consistency
            # <s> represents start of sentence, </s> represents end of sentence
            padded = ["<s>", "<s>"] + tokens + ["</s>"]
            
            # Unigrams (excluding boundary markers for vocabulary count)
            for token in tokens:
                self.unigram_counts[token] += 1
            self.total_words += len(tokens)
            
            # Bigrams
            for i in range(1, len(padded)):
                bg = (padded[i-1], padded[i])
                self.bigram_counts[bg] += 1
                
            # Trigrams
            for i in range(2, len(padded)):
                tg = (padded[i-2], padded[i-1], padded[i])
                self.trigram_counts[tg] += 1

    def calculate_unigram_prob(self, sentence: str) -> float:
        """Computes joint probability under unigram assumption."""
        tokens = sentence.lower().split()
        if not tokens or self.total_words == 0:
            return 0.0
        prob = 1.0
        for w in tokens:
            count = self.unigram_counts[w]
            prob *= (count / self.total_words)
        return prob

    def calculate_bigram_prob(self, sentence: str) -> float:
        """Computes joint probability under first-order Markov bigram assumption."""
        tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
        prob = 1.0
        for i in range(1, len(tokens)):
            prev, curr = tokens[i-1], tokens[i]
            bg = (prev, curr)
            denom = self.num_sentences if prev == "<s>" else self.unigram_counts[prev]
            if denom == 0 or self.bigram_counts[bg] == 0:
                return 0.0
            prob *= (self.bigram_counts[bg] / denom)
        return prob

    def calculate_trigram_prob(self, sentence: str) -> float:
        """Computes joint probability under second-order Markov trigram assumption."""
        tokens = ["<s>", "<s>"] + sentence.lower().split() + ["</s>"]
        prob = 1.0
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            tg = (w1, w2, w3)
            bg = (w1, w2)
            denom = self.num_sentences if (w1, w2) == ("<s>", "<s>") else self.bigram_counts[bg]
            if denom == 0 or self.trigram_counts[tg] == 0:
                return 0.0
            prob *= (self.trigram_counts[tg] / denom)
        return prob


def display_results(lm: NGramLanguageModel, test_sentences: List[str]):
    print("=" * 80)
    print("EXPERIMENT 1: PROBABILITY THEORY AND N-GRAM LANGUAGE MODELS")
    print("Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP")
    print("=" * 80)
    
    print("\nTraining Corpus (5 Documents):")
    for idx, doc in enumerate(lm.corpus, 1):
        print(f"  [{idx}] {doc}")
        
    print(f"\nCorpus Vocabulary Size: {len(lm.unigram_counts)} distinct words")
    print(f"Total Running Words    : {lm.total_words} tokens")

    print("\nTop Most Frequent Unigrams:")
    print(f"  {'Word':<18} | {'Count':<8} | {'Relative Frequency P(w)'}")
    print("  " + "-" * 50)
    for word, count in lm.unigram_counts.most_common(8):
        print(f"  {word:<18} | {count:<8} | {count / lm.total_words:.4f}")

    print("\nTop Most Frequent Bigrams:")
    print(f"  {'Bigram (w1, w2)':<28} | {'Count':<8}")
    print("  " + "-" * 40)
    for bg, count in lm.bigram_counts.most_common(8):
        bg_str = f"('{bg[0]}', '{bg[1]}')"
        print(f"  {bg_str:<28} | {count:<8}")

    print("\n" + "=" * 80)
    print("SENTENCE PROBABILITY EVALUATION")
    print("=" * 80)
    print(f"{'Target Sentence':<35} | {'Unigram P(W)':<15} | {'Bigram P(W)':<15} | {'Trigram P(W)':<15}")
    print("-" * 85)

    for s in test_sentences:
        u_p = lm.calculate_unigram_prob(s)
        b_p = lm.calculate_bigram_prob(s)
        t_p = lm.calculate_trigram_prob(s)
        print(f"{s:<35} | {u_p:<15.4e} | {b_p:<15.4e} | {t_p:<15.4e}")

    print("-" * 85)
    print("\n[Observations & Theoretical Insights]:")
    print("1. Unigram probability treats all word orders identically: P('I love AI') == P('AI love I').")
    print("2. Bigram probability captures local 2-word syntax and transitions from sentence start <s>.")
    print("3. Trigram probability models richer 3-word grammatical contexts, but suffers higher sparsity")
    print("   (unseen trigrams evaluate to 0 without smoothing).")
    print("=" * 80)


def main():
    # Training corpus from class demonstration
    sample_corpus = [
        "I love machine learning",
        "I love deep learning",
        "I study machine learning",
        "I study artificial intelligence",
        "Machine learning is interesting"
    ]

    lm = NGramLanguageModel(sample_corpus)

    test_sentences = [
        "I love machine learning",
        "I study machine learning",
        "Machine learning is interesting",
        "I love deep learning",
        "Learning machine love I"  # Permuted words: Unigram matches, Bigram/Trigram drop to 0
    ]

    display_results(lm, test_sentences)


if __name__ == "__main__":
    main()

