"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 6: CYK (Cocke-Younger-Kasami) Parsing Algorithm
File:   02. CYK Dynamic Programming Parser.py
=============================================================================

WHAT IS THE CYK ALGORITHM?
--------------------------
The CYK (Cocke-Younger-Kasami) algorithm is a dynamic programming algorithm
for parsing Context-Free Grammars. It has polynomial time complexity O(n^3 * |G|),
making it dramatically faster than naive recursive search which is exponential.

REQUIREMENT: CHOMSKY NORMAL FORM (CNF)
--------------------------------------
CYK requires the grammar to be in Chomsky Normal Form (CNF), where every rule
is either:
  1. A -> B C   (Non-terminal generating two non-terminals)
  2. A -> 'w'   (Non-terminal generating a single terminal word)

HOW THE 2D CYK CHART TABLE WORKS:
---------------------------------
For a sentence of length n (words w_0, w_1, ..., w_{n-1}):
  - Table cell (i, j) stores all non-terminals that can derive the substring
    spanning from word i to word j.
  - Base Case (Length 1, Diagonal):
      table[i][i] = { A | A -> w_i in Grammar }
  - Recursive Step (Length 2 to n):
      table[i][j] = { A | A -> B C and B in table[i][k] and C in table[k+1][j]
                          for some split point k, i <= k < j }
  - Acceptance Criterion:
      The sentence is valid if the start symbol S is in table[0][n-1].
=============================================================================
"""

# Step 1: Define Grammar in Chomsky Normal Form (CNF)
cnf_grammar = {
    "S":    [("NP", "VP")],
    "NP":   [("Det", "N"), ("Det", "Nom")],
    "Nom":  [("Adj", "N")],
    "VP":   [("V", "NP"), ("V", "PP")],
    "PP":   [("P", "NP")],
    # Terminal Lexical Rules
    "Det":  [("the",), ("a",)],
    "N":    [("student",), ("book",), ("teacher",), ("park",)],
    "Adj":  [("good",), ("new",)],
    "V":    [("reads",), ("likes",), ("sees",)],
    "P":    [("in",), ("with",)]
}

def cyk_parse(sentence_str: str, grammar_dict: dict):
    words = sentence_str.lower().split()
    n = len(words)
    
    # Initialize 2D table of sets: table[i][j] holds non-terminals spanning words[i..j]
    table = [[set() for _ in range(n)] for _ in range(n)]
    
    # Base Case: Fill diagonal cells (length 1 substrings)
    for i in range(n):
        for lhs, productions in grammar_dict.items():
            for prod in productions:
                if len(prod) == 1 and prod[0] == words[i]:
                    table[i][i].add(lhs)
                    
    # Recursive Step: Substrings of length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # Check every split point k between i and j-1
            for k in range(i, j):
                left_symbols = table[i][k]
                right_symbols = table[k + 1][j]
                
                # Check if any binary production A -> B C matches
                for lhs, productions in grammar_dict.items():
                    for prod in productions:
                        if len(prod) == 2:
                            B, C = prod
                            if B in left_symbols and C in right_symbols:
                                table[i][j].add(lhs)
                                
    return words, table

# Step 2: Test Sentences
test_cases = [
    "the student reads a book",
    "the teacher likes the park",
    "reads the student a book"   # Grammatically invalid
]

print("=" * 70)
print("EXPERIMENT 6: CYK (COCKE-YOUNGER-KASAMI) PARSING ALGORITHM")
print("=" * 70)

for sent in test_cases:
    words, chart = cyk_parse(sent, cnf_grammar)
    n = len(words)
    is_accepted = "S" in chart[0][n - 1]
    
    print(f"\nSentence: \"{sent}\"")
    print(f"Tokens  : {words}")
    print(f"Length  : {n}")
    
    print("\nUpper Triangular CYK Chart Table (Spans):")
    print("-" * 55)
    for i in range(n):
        for j in range(i, n):
            span_text = " ".join(words[i:j+1])
            non_terms = sorted(list(chart[i][j])) if chart[i][j] else "{}"
            print(f"  Span [{i}..{j}] ({span_text:<22}) --> {non_terms}")
            
    print("-" * 55)
    if is_accepted:
        print(f"VERDICT: ACCEPTED (Start symbol 'S' is in root cell [0..{n-1}])")
    else:
        print(f"VERDICT: REJECTED (Start symbol 'S' was NOT found in root cell [0..{n-1}])")

print("\n" + "=" * 70)
print("[Summary] CYK parses CNF grammars in O(n^3) using dynamic programming chart tables.")
print("=" * 70)

