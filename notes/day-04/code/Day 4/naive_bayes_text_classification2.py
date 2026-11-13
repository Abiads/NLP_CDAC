from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training dataset
documents = [
    "Congratulations you won a lottery",
    "Claim your free prize now",
    "Win money instantly",
    "Meeting scheduled for tomorrow",
    "Please submit the assignment",
    "Project discussion at 10 AM"
]

# Labels
labels = [
    "Spam",
    "Spam",
    "Spam",
    "Ham",
    "Ham",
    "Ham"
]

# Convert text into numerical features
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

# Train Naive Bayes classifier
model = MultinomialNB()

model.fit(X, labels)

# Test text
test_message = ["free lottery prize"]

# Convert test text
test_vector = vectorizer.transform(test_message)

# Predict category
prediction = model.predict(test_vector)

print("Message:", test_message[0])
print("Predicted Class:", prediction[0])
