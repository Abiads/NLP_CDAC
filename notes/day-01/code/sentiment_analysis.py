from textblob import TextBlob

print("Sentiment Analysis System")
print("-" * 30)

while True:

    text = input("\nEnter a review (or type exit): ")

    if text.lower() == "exit":
        break

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😀"

    elif polarity < 0:
        sentiment = "Negative 😞"

    else:
        sentiment = "Neutral 😐"

    print("Polarity Score:", polarity)
print("Sentiment:", sentiment)