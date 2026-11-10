
import spacy
from nltk import CFG
from nltk.parse import RecursiveDescentParser
from spacy import displacy

# ----------------------------------
# Step 1: Construct CFG
# ----------------------------------

grammar = CFG.fromstring("""
S -> NP VP

NP -> Det N
NP -> Det Adj N
NP -> 'john'

VP -> V NP
VP -> V

Det -> 'the' | 'a'

Adj -> 'big' | 'small'

N -> 'student' | 'book' | 'teacher'
N -> 'machine' | 'learning' | 'dog'

V -> 'reads' | 'explains' | 'sees'
""")

# ----------------------------------
# Step 2: Create Constituency Parser
# ----------------------------------

const_parser = RecursiveDescentParser(grammar)

# ----------------------------------
# Step 3: Load spaCy model
# ----------------------------------

nlp = spacy.load("en_core_web_sm")

# ----------------------------------
# Step 4: Sample sentences
# ----------------------------------

sentences = [
    "the student reads a book",
    "the teacher explains machine learning",
    "john sees the dog"
]

# ----------------------------------
# Step 5: Compare both parsers
# ----------------------------------

for sentence in sentences:

    print("\n================================")
    print("Sentence:", sentence)
    print("================================")

    # Constituency Parsing
    print("\nCONSTITUENCY PARSING")

    tokens = sentence.split()

    trees = list(const_parser.parse(tokens))

    if trees:
        for tree in trees:
            tree.pretty_print()
    else:
        print("No constituency parse found")

    # Dependency Parsing
    print("\nDEPENDENCY PARSING")

    doc = nlp(sentence)

    print("Word | POS | Dependency | Head")

    for token in doc:
        print(
            token.text,
            "|", token.pos_,
            "|", token.dep_,
            "|", token.head.text
        )

# ----------------------------------
# Step 6: Visualize dependency tree
# ----------------------------------

doc = nlp(sentences[0])

displacy.serve(
    doc,
    style="dep",
    port=5000
)
