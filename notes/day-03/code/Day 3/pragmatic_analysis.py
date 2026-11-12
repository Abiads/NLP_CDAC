import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required tokenizer resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Function to perform pragmatic analysis
def pragmatic_analysis(sentence):

    sentence = sentence.strip()
    lower_sentence = sentence.lower()

    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Default classification
    speech_act = "Assertive"
    intention = "The speaker is providing information."
    context = "The sentence communicates a statement."

    # Identify speech act using sentence patterns

    # Directive: Request or command
    if lower_sentence.startswith(
        ("please ", "could you ", "would you ", "can you ")
    ):
        speech_act = "Directive"
        intention = "The speaker is making a request."
        context = "The sentence asks the listener to perform an action."

    elif lower_sentence.startswith(
        ("open ", "close ", "stop ", "go ", "come ")
    ):
        speech_act = "Directive"
        intention = "The speaker is giving a command."
        context = "The sentence directs the listener to perform an action."

    # Interrogative: Question
    elif sentence.endswith("?"):
        speech_act = "Directive"
        intention = "The speaker is asking for information."
        context = "The sentence seeks a response from the listener."

    # Commissive: Promise or offer
    elif any(phrase in lower_sentence for phrase in [
        "i promise", "i will", "i shall", "i'll"
    ]):
        speech_act = "Commissive"
        intention = "The speaker is expressing a commitment."
        context = "The sentence indicates a future action or promise."

    # Expressive: Emotion or feeling
    elif any(phrase in lower_sentence for phrase in [
        "thank you", "i am sorry", "i'm sorry",
        "congratulations", "well done"
    ]):
        speech_act = "Expressive"
        intention = "The speaker is expressing a feeling or attitude."
        context = "The sentence communicates gratitude, apology, or appreciation."

    # Declarative: Declaration
    elif any(phrase in lower_sentence for phrase in [
        "i declare", "i hereby declare",
        "you are hereby", "we officially"
    ]):
        speech_act = "Declarative"
        intention = "The speaker is making a formal declaration."
        context = "The sentence announces a status or decision."

    # Display results
    print("\nSentence:", sentence)
    print("Tokens:", words)
    print("Speech Act:", speech_act)
    print("Speaker Intention:", intention)
    print("Contextual Meaning:", context)


# Sample sentences
sentences = [
    "The meeting starts at ten.",
    "Please submit your assignment.",
    "Could you open the window?",
    "I promise I will help you.",
    "Thank you for your support.",
    "I hereby declare the event open."
]

print("PRAGMATIC ANALYSIS OF NATURAL LANGUAGE")
print("------------------------------------")

for sentence in sentences:
    pragmatic_analysis(sentence)