import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
data = pd.read_csv("spam.csv")

print(data.head())


# 2. Separate input and output
X = data["text"]
y = data["label"]


# 3. Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Create model
model = MultinomialNB()


# 6. Train model
model.fit(X_train, y_train)


# 7. Make predictions
prediction = model.predict(X_test)


# 8. Evaluate model
accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction))

print("Classification Report:")
print(classification_report(y_test, prediction))


# 9. Test our own message
message = ["Congratulations! You won a free lottery prize"]

message_vector = vectorizer.transform(message)

result = model.predict(message_vector)

print("Prediction:", result[0])
