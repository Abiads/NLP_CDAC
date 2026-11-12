"""
Experiment 3: Markov Models for Text Processing and Sequence State Transitions
Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP
Lab Manual Reference: Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf (Experiment 3)

Key Objectives:
1. Model natural language sentences as first-order discrete-time Markov chains.
2. Construct state transition count tables from corpus data.
3. Compute maximum-likelihood state-transition probability distributions P(S_t | S_{t-1}).
4. Predict the next state/token given a current state using argmax transition likelihood.
5. Demonstrate multi-step Markovian text generation / state trajectory simulation.
6. Support automated benchmark testing and optional command-line queries.
Strict Zero-Emoji Policy.
"""

import sys
from collections import defaultdict, Counter


class FirstOrderMarkovModel:
    """
    First-order discrete Markov Chain model for text sequences.
    """
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.transition_probabilities = {}
        self.vocabulary = set()

    def fit(self, corpus_sentences):
        """
        Builds transition counts and calculates conditional transition probabilities.
        """
        self.transition_counts.clear()
        self.transition_probabilities.clear()
        self.vocabulary.clear()

        for sentence in corpus_sentences:
            words = sentence.lower().split()
            if not words:
                continue

            for w in words:
                self.vocabulary.add(w)

            # Record transitions between adjacent words
            for i in range(len(words) - 1):
                curr_state = words[i]
                next_state = words[i + 1]
                self.transition_counts[curr_state][next_state] += 1

        # Calculate conditional transition probabilities
        for state, next_counts in self.transition_counts.items():
            total_transitions = sum(next_counts.values())
            self.transition_probabilities[state] = {
                next_word: count / total_transitions
                for next_word, count in next_counts.items()
            }

    def predict_next_state(self, current_state):
        """
        Returns the most probable next state and full probability distribution.
        """
        state_norm = current_state.lower()
        if state_norm not in self.transition_probabilities:
            return None, {}

        probs = self.transition_probabilities[state_norm]
        best_state = max(probs, key=probs.get)
        return best_state, probs

    def generate_sequence(self, start_state, max_steps=8):
        """
        Simulates a discrete-time Markov chain walk from a starting state.
        """
        trajectory = [start_state.lower()]
        curr = start_state.lower()

        for _ in range(max_steps):
            best_next, _ = self.predict_next_state(curr)
            if not best_next:
                break
            trajectory.append(best_next)
            curr = best_next

        return " ".join(trajectory)


def run_benchmark():
    corpus = [
        "I love machine learning",
        "I love deep learning",
        "I study machine learning",
        "I study artificial intelligence",
        "Machine learning is interesting"
    ]

    print("=" * 80)
    print("EXPERIMENT 3: MARKOV MODELS FOR TEXT PROCESSING")
    print("Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP")
    print("=" * 80)

    print(f"\nTraining Corpus ({len(corpus)} sentences):")
    for idx, s in enumerate(corpus, 1):
        print(f"  [{idx}] {s}")

    model = FirstOrderMarkovModel()
    model.fit(corpus)

    print(f"\nTotal Vocabulary Size: {len(model.vocabulary)} unique states")
    print(f"Total Active States with Outgoing Transitions: {len(model.transition_probabilities)}")

    print("\n" + "=" * 80)
    print("STATE TRANSITION PROBABILITY MATRIX")
    print("=" * 80)
    print(f"{'Current State (S_t)':<18} | {'Next State (S_{t+1})':<22} | {'Probability P(S_{t+1}|S_t)'}")
    print("-" * 80)

    for state in sorted(model.transition_probabilities.keys()):
        probs = model.transition_probabilities[state]
        sorted_transitions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for next_s, p in sorted_transitions:
            print(f"{state:<18} | {next_s:<22} | {p:.4f} ({p * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("NEXT-STATE PREDICTION BENCHMARK")
    print("=" * 80)
    test_queries = ["i", "love", "machine", "study", "learning", "deep", "unknown_word"]

    for q in test_queries:
        best, dist = model.predict_next_state(q)
        if best:
            dist_str = ", ".join([f"{w}: {p:.2f}" for w, p in dist.items()])
            print(f"State: '{q:<12}' -> Predicted Next: '{best:<12}' | Distribution: {{{dist_str}}}")
        else:
            print(f"State: '{q:<12}' -> [NO OUTGOING TRANSITIONS / OUT OF VOCABULARY]")

    print("\n" + "=" * 80)
    print("MARKOVIAN STATE TRAJECTORY GENERATION")
    print("=" * 80)
    seeds = ["i", "love", "study", "machine"]
    for s in seeds:
        path = model.generate_sequence(s, max_steps=6)
        print(f"Seed: '{s}' -> State Trajectory: '{path}'")

    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS & MARKOV PROPERTY")
    print("=" * 80)
    print("1. First-Order Markov Property: P(S_{t+1} = s | S_t = s_t, ..., S_0 = s_0) = P(S_{t+1} = s | S_t = s_t).")
    print("   The future state depends exclusively on the present state, conditional on history being irrelevant.")
    print("2. Memoryless Limitation: Long-range syntactic dependencies (e.g. subject-verb agreement across clauses)")
    print("   cannot be preserved by a 1st-order chain.")
    print("3. Higher-Order Generalization: An N-gram model is an (N-1)-th order Markov model.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

    # Optional interactive CLI mode
    if len(sys.argv) > 1:
        query = sys.argv[1]
        demo_corpus = [
            "I love machine learning",
            "I love deep learning",
            "I study machine learning",
            "I study artificial intelligence",
            "Machine learning is interesting"
        ]
        m = FirstOrderMarkovModel()
        m.fit(demo_corpus)
        best_next, _ = m.predict_next_state(query)
        print(f"\n[CLI State Prediction]: Input '{query}' -> Next State '{best_next}'")

