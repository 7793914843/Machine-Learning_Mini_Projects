import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 1. Load dataset
data = pd.read_csv("heart.csv")

# 2. Display first 5 rows
print("First 5 rows:")
print(data.head())

# 3. Check dataset information
print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

# 4. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 5. Separate input and output
X = data.drop("target", axis=1)
y = data["target"]

print("\nX:")
print(X.head())

print("\ny:")
print(y.head())

# 6. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 7. Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Make predictions
prediction = model.predict(X_test)

print("\nPredictions:")
print(prediction)

print("\nActual values:")
print(y_test.values)

# 10. Calculate accuracy
accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)

# 11. Confusion Matrix
cm = confusion_matrix(y_test, prediction)

print("\nConfusion Matrix:")
print(cm)

# 12. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, prediction))

# 13. Predict a new patient
# Format:
# age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal

new_patient = [[55, 1, 1, 130, 250, 0, 1, 155, 0, 1.0, 2, 0, 2]]

new_prediction = model.predict(new_patient)

print("\nNew Patient Prediction:")

if new_prediction[0] == 1:
    print("Heart Disease: Present")
else:
    print("Heart Disease: Not Present")
