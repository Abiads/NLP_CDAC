import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

df = pd.read_csv("DisneylandReviews.csv")

sentence = "We visited the park near Disneyland."

word = "park"

tokens = word_tokenize(sentence)

sense = lesk(tokens, word)

print("Sentence:")
print(sentence)

print("\nTarget Word:")
print(word)

print("\nPredicted Sense:")
print(sense)

if sense:
    print("\nDefinition:")
    print(sense.definition())

    print("\nExample Sentences:")
    for example in sense.examples():
        print(example)