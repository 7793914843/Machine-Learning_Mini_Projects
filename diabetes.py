import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
data = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(data.head())


# 2. Separate input and output
X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Create model
model = LogisticRegression()


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


# 9. Test a new patient
new_patient = [[
    2,
    120,
    70,
    25,
    80,
    30.5,
    0.5,
    35
]]

new_patient = scaler.transform(new_patient)

result = model.predict(new_patient)

if result[0] == 1:
    print("Prediction: Diabetes")
else:
    print("Prediction: No Diabetes")
