# Iris Flower Classification

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Load the Iris dataset

iris = load_iris()


# 2. Create DataFrame

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data["target"] = iris.target


# 3. Display dataset

print("First 5 rows:")
print(data.head())


# 4. Separate input and output

X = data.drop("target", axis=1)
y = data["target"]


# 5. Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create the model

model = DecisionTreeClassifier()


# 7. Train the model

model.fit(X_train, y_train)


# 8. Make predictions

prediction = model.predict(X_test)


# 9. Calculate accuracy

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)


# 10. Confusion Matrix

cm = confusion_matrix(y_test, prediction)

print("\nConfusion Matrix:")
print(cm)


# 11. Predict a new flower

new_flower = [[
    5.1,
    3.5,
    1.4,
    0.2
]]

result = model.predict(new_flower)

print("\nPredicted Flower:")

print(iris.target_names[result[0]])
