"""
Experiment 4: Entropy, Cross-Entropy, and Information Theory in NLP
Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP
Lab Manual Reference: Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf (Experiment 4)

Key Objectives:
1. Formulate Shannon Entropy H(P) to quantify lexical uncertainty and information content in a corpus.
2. Compute Cross-Entropy H(P, Q) between an empirical test distribution P and a model distribution Q.
3. Calculate Model Perplexity PP(W) = 2^{H(P, Q)} as an intrinsic metric of predictive uncertainty.
4. Calculate Relative Entropy / Kullback-Leibler (KL) Divergence D_{KL}(P || Q) = H(P, Q) - H(P).
5. Compare entropy metrics across low-uncertainty (repetitive), high-uncertainty (uniform), and natural language texts.
Strict Zero-Emoji Policy.
"""

import math
from collections import Counter


def compute_word_probabilities(text_tokens):
    """
    Computes empirical unigram probability distribution P(w) from a token list.
    """
    counts = Counter(text_tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {w: c / total for w, c in counts.items()}


def calculate_entropy(prob_distribution):
    """
    Computes Shannon Entropy: H(P) = - sum_{w in V} P(w) * log2(P(w))
    Measured in bits per token.
    """
    entropy = 0.0
    for w, p in prob_distribution.items():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_cross_entropy(p_dist, q_dist, smoothing_epsilon=1e-5):
    """
    Computes Cross-Entropy: H(P, Q) = - sum_{w in V_P} P(w) * log2(Q(w))
    where P is the true empirical distribution and Q is the model distribution.
    """
    cross_entropy = 0.0
    for w, p in p_dist.items():
        if p > 0:
            q = q_dist.get(w, smoothing_epsilon)
            cross_entropy -= p * math.log2(q)
    return cross_entropy


def calculate_kl_divergence(p_dist, q_dist, smoothing_epsilon=1e-5):
    """
    Computes Kullback-Leibler Divergence: D_KL(P || Q) = H(P, Q) - H(P)
    Measures the extra bits needed to code samples from P using code based on Q.
    """
    h_p = calculate_entropy(p_dist)
    h_pq = calculate_cross_entropy(p_dist, q_dist, smoothing_epsilon)
    return max(0.0, h_pq - h_p)


def calculate_perplexity(cross_entropy_bits):
    """
    Computes Perplexity: PP = 2^{H(P, Q)}
    Represents the effective branching factor / equivalent weighted vocabulary size.
    """
    return math.pow(2, cross_entropy_bits)


def run_benchmark():
    # Corpus 1: General Machine Learning & AI
    corpus1 = """
    machine learning is a branch of artificial intelligence
    machine learning helps computers learn from data
    machine learning models discover statistical patterns in vast datasets
    """

    # Corpus 2: Deep Learning & Neural Networks
    corpus2 = """
    deep learning is a subset of machine learning
    deep learning uses neural networks with multi-layer architectures
    neural networks model complex non-linear representations
    """

    # Synthetic Comparative Texts
    repetitive_text = "data data data data data data data data data data"
    uniform_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

    tokens1 = corpus1.lower().split()
    tokens2 = corpus2.lower().split()
    tokens_rep = repetitive_text.lower().split()
    tokens_uni = uniform_text.lower().split()

    p1 = compute_word_probabilities(tokens1)
    p2 = compute_word_probabilities(tokens2)
    p_rep = compute_word_probabilities(tokens_rep)
    p_uni = compute_word_probabilities(tokens_uni)

    print("=" * 80)
    print("EXPERIMENT 4: ENTROPY AND INFORMATION THEORY IN NLP")
    print("Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP")
    print("=" * 80)

    print("\n[Dataset Summary]:")
    print(f"  - Corpus 1: {len(tokens1)} tokens, {len(p1)} unique vocabulary words")
    print(f"  - Corpus 2: {len(tokens2)} tokens, {len(p2)} unique vocabulary words")

    print("\n" + "=" * 80)
    print("1. SHANNON ENTROPY COMPARISON (LEXICAL UNCERTAINTY)")
    print("=" * 80)
    h1 = calculate_entropy(p1)
    h2 = calculate_entropy(p2)
    h_rep = calculate_entropy(p_rep)
    h_uni = calculate_entropy(p_uni)

    print(f"{'Corpus / Distribution':<36} | {'Entropy H(X) [bits]':<22} | {'Perplexity 2^H'}")
    print("-" * 80)
    print(f"{'Repetitive Single Word (Determinism)':<36} | {h_rep:<22.4f} | {calculate_perplexity(h_rep):.2f}")
    print(f"{'Corpus 1 (Machine Learning)':<36} | {h1:<22.4f} | {calculate_perplexity(h1):.2f}")
    print(f"{'Corpus 2 (Deep Learning)':<36} | {h2:<22.4f} | {calculate_perplexity(h2):.2f}")
    print(f"{'Uniform Distribution (10 words)':<36} | {h_uni:<22.4f} | {calculate_perplexity(h_uni):.2f}")

    print("\n" + "=" * 80)
    print("2. CROSS-ENTROPY, PERPLEXITY & KL DIVERGENCE")
    print("=" * 80)
    # Cross-Entropy evaluating Corpus 1 against Corpus 2
    ce_1_2 = calculate_cross_entropy(p1, p2, smoothing_epsilon=1e-4)
    pp_1_2 = calculate_perplexity(ce_1_2)
    kl_1_2 = calculate_kl_divergence(p1, p2, smoothing_epsilon=1e-4)

    # Self Cross-Entropy H(P, P) == H(P)
    ce_1_1 = calculate_cross_entropy(p1, p1)
    pp_1_1 = calculate_perplexity(ce_1_1)
    kl_1_1 = calculate_kl_divergence(p1, p1)

    print(f"Self-Evaluation H(P1, P1) [Theoretical Minimum]:")
    print(f"  - Cross-Entropy H(P1, P1) : {ce_1_1:.4f} bits (identically equals Shannon Entropy H(P1))")
    print(f"  - Perplexity 2^H(P1)      : {pp_1_1:.4f}")
    print(f"  - KL Divergence D_KL(P1||P1): {kl_1_1:.4f} bits (zero distance)")

    print(f"\nCross-Evaluation H(P1, Q=P2) [Distributional Mismatch]:")
    print(f"  - Cross-Entropy H(P1, P2) : {ce_1_2:.4f} bits")
    print(f"  - Cross Perplexity 2^H    : {pp_1_2:.4f}")
    print(f"  - KL Divergence D_KL(P1||P2): {kl_1_2:.4f} bits (information penalty due to vocabulary shift)")

    print("\n" + "=" * 80)
    print("3. DETAILED WORD-LEVEL SURPRISAL / INFORMATION CONTENT IN CORPUS 1")
    print("=" * 80)
    print(f"{'Token':<18} | {'Probability P(w)':<18} | {'Surprisal -log2(P) [bits]':<26} | {'Weighted Contribution -P*log2(P)'}")
    print("-" * 80)
    sorted_words = sorted(p1.items(), key=lambda x: x[1], reverse=True)
    for w, prob in sorted_words[:8]:
        surprisal = -math.log2(prob)
        contrib = prob * surprisal
        print(f"{w:<18} | {prob:<18.4f} | {surprisal:<26.4f} | {contrib:.4f}")

    print("\n" + "=" * 80)
    print("MATHEMATICAL SUMMARY")
    print("=" * 80)
    print("1. Entropy H(X): Lower entropy indicates high predictability / low randomness.")
    print("   A single constant word has H = 0 bits. A uniform distribution across N words achieves max H = log2(N).")
    print("2. Cross-Entropy H(P, Q): Upper-bounds true entropy; penalizes models Q that assign low probability")
    print("   to words frequently observed in test distribution P.")
    print("3. Perplexity PP(W): Exponential of cross-entropy; a language model with PP = 15 is as uncertain")
    print("   at each step as if choosing uniformly among 15 candidate words.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
