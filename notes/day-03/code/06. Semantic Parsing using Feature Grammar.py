r"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 8: Semantic Parsing using Feature Grammar & Lambda Calculus
File:   06. Semantic Parsing using Feature Grammar.py
=============================================================================

WHAT IS SEMANTIC PARSING?
-------------------------
Semantic parsing is the task of converting a natural language sentence into a
machine-readable, formal semantic / logical representation (such as First-Order
Logic, Lambda Calculus, or SQL).

THE PRINCIPLE OF COMPOSITIONALITY (FREGE'S PRINCIPLE)
------------------------------------------------------
The meaning of a complex expression is determined by the meanings of its
syntactic parts and the rules used to combine them:
  Meaning(S) = Meaning(VP)(Meaning(NP))

LAMBDA CALCULUS (BETA-REDUCTION):
---------------------------------
1. Entity constants (Arguments):
     NP[SEM=<john>] -> 'John'
     NP[SEM=<mary>] -> 'Mary'
2. Intransitive Verbs (One-place Predicates):
     VP[SEM=<\x.walk(x)>] -> 'walks'
     Application: (\x.walk(x))(john)  ==[beta-reduction]==> walk(john)
3. Transitive Verbs (Two-place Predicates):
     TV[SEM=<\y x.love(x, y)>] -> 'loves'
     VP[SEM=<?tv(?obj)>] -> TV[SEM=?tv] NP[SEM=?obj]
     Application: (\y x.love(x, y))(mary)  ==> \x.love(x, mary)
     Then applied to subject 'john'         ==> love(john, mary)

=============================================================================
"""

import nltk
try:
    from nltk import FeatureGrammar
except ImportError:
    from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureChartParser


# Step 1: Define Compositional Semantic Grammar in Lambda Calculus
semantic_grammar_text = r"""
% start S

# Sentence rule combining Subject (NP) and Predicate (VP)
S[SEM=<?vp(?np)>] -> NP[SEM=?np] VP[SEM=?vp]

# Intransitive Verb Phrase (1-place predicate)
VP[SEM=?v] -> IV[SEM=?v]

# Transitive Verb Phrase (2-place predicate applied to object)
VP[SEM=<?tv(?obj)>] -> TV[SEM=?tv] NP[SEM=?obj]

# Proper Noun / Entity Semantics
NP[SEM=<john>] -> 'John'
NP[SEM=<mary>] -> 'Mary'
NP[SEM=<alice>] -> 'Alice'
NP[SEM=<bob>] -> 'Bob'
NP[SEM=<the_dog>] -> 'the' 'dog'
NP[SEM=<the_cat>] -> 'the' 'cat'

# Intransitive Verbs (\x. P(x))
IV[SEM=<\x.walk(x)>] -> 'walks'
IV[SEM=<\x.run(x)>] -> 'runs'
IV[SEM=<\x.bark(x)>] -> 'barks'
IV[SEM=<\x.sleep(x)>] -> 'sleeps'

# Transitive Verbs (\y x. P(x, y))
TV[SEM=<\y x.love(x,y)>] -> 'loves'
TV[SEM=<\y x.feed(x,y)>] -> 'feeds'
TV[SEM=<\y x.chase(x,y)>] -> 'chases'
TV[SEM=<\y x.see(x,y)>] -> 'sees'
"""

grammar = FeatureGrammar.fromstring(semantic_grammar_text)
parser = FeatureChartParser(grammar)


def parse_semantics(sentence: str):
    """
    Parses a natural language sentence and extracts its logical representation.
    """
    tokens = sentence.split()
    print("-" * 75)
    print(f"Input Sentence : \"{sentence}\"")
    print(f"Tokens         : {tokens}")
    print("-" * 75)
    
    try:
        parses = list(parser.parse(tokens))
        if parses:
            for idx, tree in enumerate(parses, 1):
                logical_form = tree.label()["SEM"]
                print(f"Parse Derivation #{idx}:")
                print(f"  Syntactic Tree          : {tree}")
                print(f"  Formal Semantic Formula : {logical_form}")
                print(f"  Interpretation          : Predicate = '{logical_form.variable if hasattr(logical_form, 'variable') else logical_form}'")
        else:
            print("  Status: No valid semantic parse found under grammar.")
    except ValueError as e:
        print(f"  Parsing Error: {e}")
    print()


def main():
    print("=" * 75)
    print("EXPERIMENT 8: SEMANTIC PARSING (NATURAL LANGUAGE TO FIRST-ORDER LOGIC)")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & Semantic Representations")
    print("=" * 75)
    
    test_sentences = [
        # Intransitive sentences
        "John walks",
        "Mary runs",
        "the dog barks",
        "Alice sleeps",
        # Transitive sentences
        "John loves Mary",
        "Mary feeds the dog",
        "the dog chases the cat",
        "Alice sees Bob"
    ]
    
    for sentence in test_sentences:
        parse_semantics(sentence)

    print("=" * 75)
    print("[Summary] Semantic Parsing maps syntactic parse trees directly into")
    print("compositional First-Order Logic formulas using Lambda Calculus:")
    print("  - Intransitive: \\x. P(x) + NP(c)  ==> P(c)")
    print("  - Transitive  : \\y \\x. P(x, y) + NP(c1) + NP(c2) ==> P(c2, c1)")
    print("=" * 75)


if __name__ == "__main__":
    main()
