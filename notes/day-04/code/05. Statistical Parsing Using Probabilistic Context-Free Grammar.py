"""
Experiment 5: Statistical Parsing Using Probabilistic Context-Free Grammar (PCFG)
Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP
Lab Manual Reference: Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf (Experiment 5)

Key Objectives:
1. Construct formal Probabilistic Context-Free Grammars (PCFG) where rule probabilities sum to 1.0 per non-terminal.
2. Implement statistical chart parsing using the Viterbi dynamic programming algorithm.
3. Compute the joint derivation probability of competing parse trees: P(T) = prod_{r in T} P(r).
4. Resolve classic syntactic ambiguities (such as Prepositional Phrase attachment) statistically.
5. Provide pretty-printed constituency trees and detailed derivation probability accounting.
Strict Zero-Emoji Policy.
"""

import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser, pchart


def build_baseline_pcfg():
    """
    Constructs a baseline PCFG covering transitive, intransitive, and PP-modifying structures.
    """
    grammar_str = """
    S  -> NP VP             [1.0]

    NP -> Pro               [0.25]
    NP -> Det N             [0.35]
    NP -> NP PP             [0.25]
    NP -> N                 [0.15]

    VP -> V NP              [0.50]
    VP -> VP PP             [0.30]
    VP -> V                 [0.20]

    PP -> P NP              [1.0]

    Pro -> 'I'              [0.60]
    Pro -> 'they'           [0.40]

    Det -> 'the'            [0.70]
    Det -> 'a'              [0.30]

    N   -> 'astronomer'     [0.30]
    N   -> 'telescope'      [0.30]
    N   -> 'stars'          [0.20]
    N   -> 'students'       [0.20]

    V   -> 'saw'            [0.60]
    V   -> 'study'          [0.25]
    V   -> 'learn'          [0.15]

    P   -> 'with'           [0.70]
    P   -> 'through'        [0.30]
    """
    return PCFG.fromstring(grammar_str)


def parse_and_evaluate(grammar, sentence_str):
    """
    Parses a sentence using Viterbi dynamic programming and returns top trees with probabilities.
    """
    tokens = sentence_str.split()
    parser = ViterbiParser(grammar)
    trees = list(parser.parse(tokens))

    # Also extract all parses using probabilistic chart parser for comparison
    chart_parser = pchart.InsideChartParser(grammar)
    all_trees = list(chart_parser.parse(tokens))
    # Sort all parses by descending probability
    all_trees = sorted(all_trees, key=lambda t: t.prob(), reverse=True)

    return trees, all_trees


def run_benchmark():
    grammar = build_baseline_pcfg()

    print("=" * 80)
    print("EXPERIMENT 5: STATISTICAL PARSING USING PCFG (VITERBI ALGORITHM)")
    print("Course: CDAC PGCP-AI - Session 5 Statistical Approaches in NLP")
    print("=" * 80)

    print("\n[PCFG Grammar Rules Summary]:")
    for nonterminal in sorted(set(p.lhs() for p in grammar.productions()), key=lambda x: str(x)):
        rules = [p for p in grammar.productions() if p.lhs() == nonterminal]
        total_p = sum(p.prob() for p in rules)
        print(f"  - Nonterminal {nonterminal}: {len(rules)} productions (sum of probabilities = {total_p:.4f})")

    # Benchmark 1: Simple Transitive Sentence
    test_sent_1 = "I study stars"
    print("\n" + "=" * 80)
    print(f"BENCHMARK 1: UNAMBIGUOUS SENTENCE -> '{test_sent_1}'")
    print("=" * 80)

    viterbi_trees_1, all_trees_1 = parse_and_evaluate(grammar, test_sent_1)
    if viterbi_trees_1:
        top_tree = viterbi_trees_1[0]
        print("\nMost Probable Parse Tree (Viterbi):")
        top_tree.pretty_print()
        print(f"Joint Derivation Probability P(T): {top_tree.prob():.6e}")
    else:
        print("No valid parse found.")

    # Benchmark 2: Syntactically Ambiguous Prepositional Phrase Attachment
    test_sent_2 = "I saw the astronomer with the telescope"
    print("\n" + "=" * 80)
    print(f"BENCHMARK 2: PP ATTACHMENT AMBIGUITY -> '{test_sent_2}'")
    print("=" * 80)

    viterbi_trees_2, all_trees_2 = parse_and_evaluate(grammar, test_sent_2)

    print(f"\nTotal Candidate Parse Trees Derived: {len(all_trees_2)}")
    for idx, t in enumerate(all_trees_2, 1):
        print(f"\n--- Candidate Parse Tree #{idx} [Probability: {t.prob():.6e}] ---")
        t.pretty_print()

        # Identify attachment type
        t_str = str(t)
        if "(VP (VP" in t_str or "(VP (V saw) (NP" in t_str and "PP" in t_str:
            if "(NP (NP" in t_str:
                attachment = "NP Attachment: 'the astronomer with the telescope' (Object Modifier)"
            else:
                attachment = "VP Attachment: 'saw ... with the telescope' (Instrumental Modifier)"
        else:
            attachment = "Alternative Structural Derivation"
        print(f"Syntactic Interpretation: {attachment}")

    if viterbi_trees_2:
        print("\n" + "-" * 80)
        print("VITERBI STATISTICAL DECISION:")
        print(f"Selected Optimal Parse Tree Probability: {viterbi_trees_2[0].prob():.6e}")
        print("The PCFG resolves the ambiguity by favoring the derivation that yields maximum joint likelihood.")

    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS")
    print("=" * 80)
    print("1. PCFG Rule Normalization: For every non-terminal symbol A, sum_{beta} P(A -> beta) = 1.0.")
    print("2. Tree Probability Independence Assumption: P(T) = prod_{A -> beta in T} P(A -> beta).")
    print("   Assumes rule expansions are conditionally independent given the non-terminal symbol.")
    print("3. Viterbi Dynamic Programming: Extends CKY parsing by computing maximum sub-span probabilities,")
    print("   finding the argmax parse in O(n^3 * |R|) time complexity.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
