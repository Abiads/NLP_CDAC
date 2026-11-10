# !pip install -q svgling

import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser
from IPython.display import display

# Step 1: Define the Context-Free Grammar
grammar = CFG.fromstring("""
S -> NP VP

NP -> Det N
NP -> Det Adj N
NP -> 'John'

VP -> V NP
VP -> V

Det -> 'the' | 'a'
Adj -> 'big' | 'small'

N -> 'dog' | 'cat' | 'ball'
V -> 'sees' | 'chases' | 'runs'
""")

# Step 2: Create the parser
parser = RecursiveDescentParser(grammar)

# Step 3: Define the input sentences
sentences = [
    "the dog runs",
    "the big cat sees a dog",
    "John chases the ball"
]

# Step 4: Parse each sentence
for sentence in sentences:

    print("\nSentence:", sentence)

    # Convert sentence into a list of words
    tokens = sentence.split()

    # Generate parse trees
    parse_trees = list(parser.parse(tokens))

    if parse_trees:
        for tree in parse_trees:

            print("\nParse Tree (Text-based):")
            tree.pretty_print()

            # Display tree graphically inside Jupyter Notebook
            display(tree)

    else:
        print("No valid parse tree found.")