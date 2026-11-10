import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet
# Download required WordNet resource
nltk.download('wordnet')
nltk.download('omw-1.4')
# Input sentences
sentence1 = "The fisherman sat on the bank of the river."
sentence2 = "She deposited money in the bank."
# Ambiguous word
word = "bank"
# Apply Lesk algorithm
sense1 = lesk(sentence1.split(), word)
sense2 = lesk(sentence2.split(), word)
# Display results
print("WORD SENSE DISAMBIGUATION")
print("=" * 50)
print("\nSentence 1:")
print(sentence1)
print("\nSentence 2:")
print(sentence2)
print("\nAmbiguous Word:", word)
print("\nMeaning in Sentence 1:")
if sense1:
    print(sense1.definition())
    print("Synset:", sense1.name())
else:
    print("No meaning identified")
print("\nMeaning in Sentence 2:")
if sense2:
    print(sense2.definition())
    print("Synset:", sense2.name())
else:
    print("No meaning identified")
