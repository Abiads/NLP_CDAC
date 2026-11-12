
import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

# STEP 1: Define Probabilistic Context-Free Grammar

grammar = PCFG.fromstring("""
S -> NP VP [1.0]

NP -> Det N [0.7]
NP -> NP PP [0.3]

VP -> V NP [0.7]
VP -> VP PP [0.3]

PP -> P NP [1.0]

Det -> 'the' [0.6]
Det -> 'a' [0.4]

N -> 'man' [0.5]
N -> 'telescope' [0.5]

V -> 'saw' [1.0]

P -> 'with' [1.0]
""")

# STEP 2: Create an input sentence

sentence = "the man saw the man with a telescope"

# STEP 3: Tokenize the sentence

tokens = sentence.lower().split()

print("INPUT SENTENCE:")
print(sentence)

print("\nTOKENS:")
print(tokens)

# STEP 4: Create Viterbi parser

parser = ViterbiParser(grammar)

# STEP 5: Generate the most probable parse tree

parses = list(parser.parse(tokens))

if parses:

    best_parse = parses[0]

    print("\nMOST PROBABLE PARSE TREE:")
    print(best_parse)

    print("\nTREE DIAGRAM:")
    best_parse.pretty_print()

    print("\nPROBABILITY OF PARSE TREE:")
    print(best_parse.prob())

else:
    print("\nNo valid parse tree found.")
