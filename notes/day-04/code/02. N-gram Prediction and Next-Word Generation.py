"""
Experiment 2: Discrete-Time Language Models and N-gram Next-Word Prediction
Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP
Lab Manual Reference: Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf (Experiment 2)

Key Objectives:
1. Construct discrete-time Markovian bigram and trigram transition mappings from text corpora.
2. Calculate empirical conditional next-word probability distributions P(w_t | w_{t-1}) and P(w_t | w_{t-2}, w_{t-1}).
3. Predict the most probable next token given context words.
4. Implement greedy multi-step sentence generation / autoregressive decoding.
5. Provide both automated test suites and optional user input handling.
Strict Zero-Emoji Policy.
"""

import sys
from collections import defaultdict, Counter


def build_ngram_models(corpus_sentences):
    """
    Constructs bigram and trigram transition dictionaries with conditional counts.
    """
    bigram_transitions = defaultdict(Counter)
    trigram_transitions = defaultdict(Counter)

    for sentence in corpus_sentences:
        tokens = ["<s>"] + sentence.lower().split() + ["</s>"]

        # Build Bigrams: (w_{t-1}) -> w_t
        for i in range(len(tokens) - 1):
            prev_word = tokens[i]
            next_word = tokens[i + 1]
            bigram_transitions[prev_word][next_word] += 1

        # Build Trigrams: (w_{t-2}, w_{t-1}) -> w_t
        for i in range(len(tokens) - 2):
            context = (tokens[i], tokens[i + 1])
            target = tokens[i + 2]
            trigram_transitions[context][target] += 1

    return bigram_transitions, trigram_transitions


def predict_next_word(context_word, bigram_transitions):
    """
    Predicts the most probable next word and returns the distribution.
    """
    word_norm = context_word.lower()
    if word_norm not in bigram_transitions:
        return None, []

    counter = bigram_transitions[word_norm]
    total = sum(counter.values())
    ranked = sorted([(w, count / total, count) for w, count in counter.items()],
                    key=lambda x: x[1], reverse=True)
    best_word = ranked[0][0]
    return best_word, ranked


def generate_sequence(seed_word, bigram_transitions, max_len=10):
    """
    Autoregressively generates a word sequence using greedy maximum-likelihood transitions.
    """
    current = seed_word.lower()
    generated = [current]

    for _ in range(max_len):
        best_next, _ = predict_next_word(current, bigram_transitions)
        if not best_next or best_next == "</s>":
            break
        generated.append(best_next)
        current = best_next

    return " ".join(generated)


def run_benchmark():
    corpus = [
        "I love machine learning",
        "I love deep learning",
        "I study machine learning",
        "I study artificial intelligence",
        "Machine learning is interesting",
        "Deep learning is a subset of machine learning",
        "Artificial intelligence is transforming industries"
    ]

    print("=" * 80)
    print("EXPERIMENT 2: DISCRETE-TIME LANGUAGE MODELS & N-GRAM PREDICTION")
    print("Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP")
    print("=" * 80)

    print(f"\nTraining Corpus ({len(corpus)} sentences):")
    for idx, s in enumerate(corpus, 1):
        print(f"  [{idx}] {s}")

    bigram_transitions, trigram_transitions = build_ngram_models(corpus)

    print("\n" + "=" * 80)
    print("EMPIRICAL CONDITIONAL NEXT-WORD DISTRIBUTIONS")
    print("=" * 80)

    test_words = ["machine", "i", "love", "study", "deep", "artificial", "learning", "unknown_token"]

    print(f"{'Input Word':<16} | {'Top Prediction':<16} | {'Probability':<12} | {'All Candidate Transitions'}")
    print("-" * 80)

    for word in test_words:
        best, dist = predict_next_word(word, bigram_transitions)
        if best is not None:
            top_prob = dist[0][1]
            cands_str = ", ".join([f"{w}: {p:.2f} ({c})" for w, p, c in dist])
            print(f"{word:<16} | {best:<16} | {top_prob:<12.4f} | {cands_str}")
        else:
            print(f"{word:<16} | {'[OUT OF VOCAB]':<16} | {'0.0000':<12} | None (word not observed in training corpus)")

    print("\n" + "=" * 80)
    print("GREEDY AUTOREGRESSIVE SEQUENCE COMPLETION")
    print("=" * 80)
    seeds = ["i", "machine", "deep", "artificial"]
    for s in seeds:
        completion = generate_sequence(s, bigram_transitions, max_len=8)
        print(f"Seed: '{s}' -> Completed: '{completion}'")

    print("\n" + "=" * 80)
    print("MATHEMATICAL & THEORETICAL SUMMARY")
    print("=" * 80)
    print("1. Bigram Markov Assumption: P(w_t | w_1...w_{t-1}) approx P(w_t | w_{t-1}).")
    print("2. Maximum Likelihood Estimate: P(w_t | w_{t-1}) = Count(w_{t-1}, w_t) / Count(w_{t-1}).")
    print("3. Discrete-Time State Transition: Words act as discrete states in a finite Markov chain.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

    # Optional interactive mode when arguments are provided or standard input is active
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        corpus = [
            "I love machine learning",
            "I love deep learning",
            "I study machine learning",
            "I study artificial intelligence",
            "Machine learning is interesting"
        ]
        b_trans, _ = build_ngram_models(corpus)
        best, dist = predict_next_word(user_query, b_trans)
        print(f"\n[CLI Query]: '{user_query}' -> Top Prediction: '{best}'")

