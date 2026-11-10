
import nltk
try:
    from nltk import FeatureGrammar
except ImportError:
    from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureChartParser

# Step 1: Define grammar with semantic rules
grammar_text = r"""
% start S

S[SEM=<?vp(?np)>] -> NP[SEM=?np] VP[SEM=?vp]

NP[SEM=<john>] -> 'John'
NP[SEM=<mary>] -> 'Mary'
NP[SEM=<the_dog>] -> 'the' 'dog'

VP[SEM=<\x.walk(x)>] -> 'walks'
VP[SEM=<\x.runs(x)>] -> 'runs'

VP[SEM=<\x.barks(x)>] -> 'barks'
"""

# Step 2: Load the grammar
grammar = FeatureGrammar.fromstring(grammar_text)

# Step 3: Create the parser
parser = FeatureChartParser(grammar)

# Step 4: Define sample sentences
sentences = [
    "John walks",
    "Mary runs",
    "the dog barks"
]

# Step 5: Parse and generate logical representations
for sentence in sentences:

    print("\nSentence:", sentence)

    tokens = sentence.split()

    parses = list(parser.parse(tokens))

    if parses:

        for tree in parses:

            print("\nParse Tree:")
            print(tree)

            logical_representation = tree.label()["SEM"]

            print("\nLogical Representation:")
            print(logical_representation)

    else:
        print("No valid parse found.")
