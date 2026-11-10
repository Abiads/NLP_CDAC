
import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser

# Step 1: Construct the CFG
grammar = CFG.fromstring("""
S -> NP VP

NP -> Det N
NP -> Det Adj N
NP -> 'john'

VP -> V NP
VP -> V

Det -> 'the' | 'a'

Adj -> 'big' | 'small'

N -> 'dog' | 'cat' | 'ball'

V -> 'sees' | 'chases' | 'runs'
""")

# Step 2: Create the parser
parser = RecursiveDescentParser(grammar)

# Step 3: Define sentences
sentences = [
    "the dog runs",
    "the big cat sees a dog",
    "john chases the ball"
]

# Step 4: Parse each sentence
for sentence in sentences:

    print("\nSentence:", sentence)

    tokens = sentence.lower().split()

    parse_trees = list(parser.parse(tokens))

    # Step 5: Check whether a parse tree exists
    if parse_trees:

        print("Sentence is VALID")

        # Step 6: Generate constituency parse trees
        for tree in parse_trees:

            print("\nConstituency Parse Tree:")
            tree.pretty_print()

    else:

        print("Sentence is INVALID")
        print("The sentence does not follow the grammar.")
