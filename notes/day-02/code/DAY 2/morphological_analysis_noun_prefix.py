import nltk
from nltk.corpus import wordnet as wn

# Download required NLTK resources
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

# Prefixes
prefixes = [
    "un", "re", "dis", "mis",
    "pre", "in", "im", "non"
]

# Suffixes
suffixes = [
    "ing", "ed", "ly", "ness",
    "ment", "er", "ful", "less",
    "es", "s"
]


def is_valid_root(root):

    # Check whether root is a recognized English word
    return bool(wn.synsets(root))


def morphological_analysis(word):

    original_word = word.strip()

    # POS tagging
    pos_tag = nltk.pos_tag([original_word])[0][1]

    print("\nWord:", original_word)
    print("POS Tag:", pos_tag)

    # Proper nouns should not be blindly split
    if pos_tag == "NNP" or pos_tag == "NNPS":

        print("Prefix: None")
        print("Root Word:", original_word)
        print("Suffix: None")
        print("Note: Proper noun; no morphological split applied.")
        return

    word = original_word.lower()

    prefix = ""
    suffix = ""
    root = word

    # Identify prefix only if remaining root is valid
    for p in sorted(prefixes, key=len, reverse=True):

        if word.startswith(p):

            candidate = word[len(p):]

            if is_valid_root(candidate):

                prefix = p
                root = candidate
                break

    # Identify suffix only if remaining root is valid
    for s in sorted(suffixes, key=len, reverse=True):

        if root.endswith(s):

            candidate = root[:-len(s)]

            if is_valid_root(candidate):

                suffix = s
                root = candidate
                break

    print("Prefix:", prefix if prefix else "None")
    print("Root Word:", root)
    print("Suffix:", suffix if suffix else "None")


# Main program
word = input("Enter a word: ")

morphological_analysis(word)
