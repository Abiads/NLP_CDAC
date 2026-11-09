import ast

# An artificial language: one fixed parse
tree = ast.parse("x = a + b * c")
print(ast.dump(tree, indent=2))

# A natural-language sentence: just a string
sent = "I saw the man with the telescope"
print(sent.split())