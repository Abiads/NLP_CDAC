import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

# Define PCFG Grammar
grammar = PCFG.fromstring("""
S -> NP VP [1.0]

NP -> 'I' [0.5]
NP -> 'students' [0.5]

VP -> V NP [0.6]
VP -> V [0.4]

V -> 'study' [0.5]
V -> 'learn' [0.5]
""")

# Create Parser
parser = ViterbiParser(grammar)

# Input sentence
sentence = "I study students".split()

# Parse sentence
print("Most Probable Parse Tree:\n")

for tree in parser.parse(sentence):
    print(tree)
    print("\nProbability =", tree.prob())