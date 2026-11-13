
# TEXT CLASSIFICATION USING SUPPORT VECTOR MACHINE

# STEP 1: Import required libraries

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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


# STEP 3: Define category labels

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


# STEP 5: Convert text into TF-IDF features

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


print("TF-IDF Feature Matrix:")
print(X_train_tfidf.toarray())

print("\nVocabulary:")
print(tfidf.get_feature_names_out())


# STEP 6: Create and train SVM classifier

svm_model = SVC(kernel='linear')

svm_model.fit(X_train_tfidf, y_train)


# STEP 7: Predict categories of test documents

y_pred = svm_model.predict(X_test_tfidf)


# STEP 8: Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label="Technology",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label="Technology",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label="Technology",
    zero_division=0
)


# STEP 9: Display evaluation results

print("\n===== SVM CLASSIFICATION RESULTS =====")

print("Accuracy:", round(accuracy, 4))

print("Precision:", round(precision, 4))

print("Recall:", round(recall, 4))

print("F1-Score:", round(f1, 4))


# STEP 10: Classification report

print("\nCLASSIFICATION REPORT:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# STEP 11: Confusion matrix

print("\nCONFUSION MATRIX:")

print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Sports", "Technology"]
))


# STEP 12: Predict category of new documents

new_documents = [
    "Artificial intelligence helps computers learn",
    "The cricket team won the tournament",
    "Python programming is used in software development"
]

new_tfidf = tfidf.transform(new_documents)

predictions = svm_model.predict(new_tfidf)

print("\n===== NEW DOCUMENT PREDICTIONS =====")

for text, category in zip(new_documents, predictions):

    print("\nDocument:", text)

    print("Predicted Category:", category)
