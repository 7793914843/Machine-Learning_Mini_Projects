import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
data = pd.read_csv("loan_data.csv")

print("First 5 rows:")
print(data.head())


# 2. Convert Approved into numbers
data["Approved"] = data["Approved"].map({
    "Yes": 1,
    "No": 0
})


# 3. Select input features
X = data[[
    "Income",
    "Age",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
]]


# 4. Select target
y = data["Approved"]


# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create the model
model = DecisionTreeClassifier(random_state=42)


# 7. Train the model
model.fit(X_train, y_train)


# 8. Make predictions
prediction = model.predict(X_test)


# 9. Check accuracy
accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)


# 10. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))


# 11. Classification report
print("\nClassification Report:")
print(classification_report(y_test, prediction))


# 12. Predict a new applicant
new_applicant = [[
    60000,     # Income
    30,        # Age
    200000,    # Loan Amount
    740,       # Credit Score
    5          # Employment Years
]]

result = model.predict(new_applicant)


if result[0] == 1:
    print("\nLoan Status: Approved")
else:
    print("\nLoan Status: Not Approved")
