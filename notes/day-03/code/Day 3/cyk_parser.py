
# CYK Parsing Algorithm

# Grammar in Chomsky Normal Form (CNF)
grammar = {
    "S":  [("NP", "VP")],
    "NP": [("Det", "N")],
    "VP": [("V", "NP")],
    "Det": [("the",), ("a",)],
    "N":   [("student",), ("book",)],
    "V":   [("reads",)]
}

# Input sentence
sentence = "the student reads a book"

# Split sentence into words
words = sentence.lower().split()

n = len(words)

# Create CYK table
table = [[set() for j in range(n)] for i in range(n)]

# Step 1: Fill the diagonal with terminal rules
for i in range(n):
    for lhs, rules in grammar.items():
        for rule in rules:
            if len(rule) == 1 and rule[0] == words[i]:
                table[i][i].add(lhs)

# Step 2: Fill the table for longer word sequences
for length in range(2, n + 1):

    for start in range(n - length + 1):

        end = start + length - 1

        # Try every possible split
        for split in range(start, end):

            left = table[start][split]
            right = table[split + 1][end]

            # Check grammar rules
            for lhs, rules in grammar.items():

                for rule in rules:

                    if len(rule) == 2:

                        B, C = rule

                        if B in left and C in right:
                            table[start][end].add(lhs)

# Step 3: Display CYK table
print("CYK Parsing Table")
print("----------------")

for i in range(n):
    for j in range(n):
        if j >= i:
            print(f"({i},{j}): {table[i][j]}")

# Step 4: Check whether sentence belongs to grammar
print("\nSentence:", sentence)

if "S" in table[0][n - 1]:
    print("Result: Sentence is ACCEPTED by the grammar.")
else:
    print("Result: Sentence is REJECTED by the grammar.")
