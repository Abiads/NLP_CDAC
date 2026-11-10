import pandas as pd

def morphological_synthesis(root_word, feature):

    if feature == "plural":
        if root_word.endswith("y"):
            return root_word[:-1] + "ies"
        else:
            return root_word + "s"

    elif feature == "past":
        if root_word.endswith("e"):
            return root_word + "d"
        else:
            return root_word + "ed"

    elif feature == "present_participle":
        if root_word.endswith("e"):
            return root_word[:-1] + "ing"
        else:
            return root_word + "ing"

    return root_word

df = pd.read_csv("DisneylandReviews.csv")

words = ["ride", "visit", "park", "play", "travel"]

print("Morphological Synthesis")
print("="*40)

for word in words:

    print("\nRoot Word:", word)

    print("Plural:",
          morphological_synthesis(word, "plural"))

    print("Past Tense:",
          morphological_synthesis(word, "past"))

    print("Present Participle:",
          morphological_synthesis(word,
                                  "present_participle"))