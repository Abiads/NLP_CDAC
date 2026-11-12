
# TEXT CATEGORIZATION USING MACHINE LEARNING

# STEP 1: Import required libraries

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# STEP 2: Create sample text documents

documents = [
    "India won the cricket match",
    "The football team scored a goal",
    "The player scored a century in cricket",
    "The team won the football tournament",
    "Cricket is a popular sport",
    "The coach trained the football players",
    "The batsman hit a six in the match",
    "The goalkeeper saved the penalty",
    "The tennis player won the match",
    "The basketball team scored points",

    "Artificial intelligence is transforming technology",
    "Machine learning is used in healthcare",
    "Python is a popular programming language",
    "Deep learning is a branch of artificial intelligence",
    "Computer networks connect different devices",
    "Data science uses machine learning algorithms",
    "Cloud computing provides online services",
    "Cybersecurity protects computer systems",
    "Robotics uses artificial intelligence",
    "Software development uses programming languages"
]


# STEP 3: Define document categories

labels = [
    "Sports", "Sports", "Sports", "Sports", "Sports",
    "Sports", "Sports", "Sports", "Sports", "Sports",

    "Technology", "Technology", "Technology",
    "Technology", "Technology", "Technology",
    "Technology", "Technology", "Technology",
    "Technology"
]


# STEP 4: Divide data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


# STEP 5: BAG OF WORDS FEATURE EXTRACTION

print("\n===== BAG OF WORDS =====")

bow_vectorizer = CountVectorizer()

X_train_bow = bow_vectorizer.fit_transform(X_train)

X_test_bow = bow_vectorizer.transform(X_test)

print("Vocabulary:")
print(bow_vectorizer.get_feature_names_out())

print("\nTraining Matrix:")
print(X_train_bow.toarray())


# STEP 6: Train Naive Bayes classifier using BoW

bow_model = MultinomialNB()

bow_model.fit(X_train_bow, y_train)

bow_predictions = bow_model.predict(X_test_bow)

bow_accuracy = accuracy_score(y_test, bow_predictions)

print("\nBoW Accuracy:", bow_accuracy)

print("\nBoW Classification Report:")
print(classification_report(y_test, bow_predictions))


# STEP 7: TF-IDF FEATURE EXTRACTION

print("\n===== TF-IDF =====")

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(X_train_tfidf.toarray())


# STEP 8: Train Naive Bayes classifier using TF-IDF

tfidf_model = MultinomialNB()

tfidf_model.fit(X_train_tfidf, y_train)

tfidf_predictions = tfidf_model.predict(X_test_tfidf)

tfidf_accuracy = accuracy_score(y_test, tfidf_predictions)

print("\nTF-IDF Accuracy:", tfidf_accuracy)

print("\nTF-IDF Classification Report:")
print(classification_report(y_test, tfidf_predictions))


# STEP 9: Compare both methods

print("\n===== PERFORMANCE COMPARISON =====")

print("Bag of Words Accuracy:", bow_accuracy)

print("TF-IDF Accuracy:", tfidf_accuracy)


# STEP 10: Predict category of a new document

new_document = [
    "Artificial intelligence helps computers learn",
    "The cricket team won the tournament"
]

print("\n===== NEW DOCUMENT PREDICTION =====")

# Using Bag of Words
new_bow = bow_vectorizer.transform(new_document)

bow_result = bow_model.predict(new_bow)

# Using TF-IDF
new_tfidf = tfidf_vectorizer.transform(new_document)

tfidf_result = tfidf_model.predict(new_tfidf)

for i, text in enumerate(new_document):

    print("\nDocument:", text)

    print("BoW Prediction:", bow_result[i])

    print("TF-IDF Prediction:", tfidf_result[i])
