# ==========================================
# CUSTOMER CHURN PREDICTION
# ==========================================

# 1. Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==========================================
# 2. Load Dataset
# ==========================================

data = pd.read_csv("customer_churn.csv")

print("First 5 rows:")
print(data.head())


# ==========================================
# 3. Check Dataset
# ==========================================

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())


# ==========================================
# 4. Convert Text into Numbers
# ==========================================

encoder = LabelEncoder()

data["Contract"] = encoder.fit_transform(data["Contract"])

data["InternetService"] = encoder.fit_transform(
    data["InternetService"]
)

data["PaymentMethod"] = encoder.fit_transform(
    data["PaymentMethod"]
)

data["TechSupport"] = encoder.fit_transform(
    data["TechSupport"]
)

data["OnlineSecurity"] = encoder.fit_transform(
    data["OnlineSecurity"]
)

# Convert Churn
data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})


print("\nAfter Encoding:")
print(data.head())


# ==========================================
# 5. Separate Input and Output
# ==========================================

X = data.drop("Churn", axis=1)

y = data["Churn"]


print("\nInput values:")
print(X.head())

print("\nOutput values:")
print(y.head())


# ==========================================
# 6. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# ==========================================
# 7. Create Model
# ==========================================

model = DecisionTreeClassifier(
    random_state=42
)


# ==========================================
# 8. Train Model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 9. Make Predictions
# ==========================================

prediction = model.predict(X_test)


# ==========================================
# 10. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy:")
print(accuracy)


# ==========================================
# 11. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)


# ==========================================
# 12. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)


# ==========================================
# 13. Predict a New Customer
# ==========================================

new_customer = [[
    30,      # Age
    4,       # Tenure
    90.0,    # MonthlyCharges
    360.0,   # TotalCharges
    0,       # Contract
    1,       # InternetService
    2,       # PaymentMethod
    0,       # TechSupport
    0        # OnlineSecurity
]]


result = model.predict(new_customer)


if result[0] == 1:
    print("\nPrediction: Customer is likely to CHURN")
else:
    print("\nPrediction: Customer is likely to STAY")
