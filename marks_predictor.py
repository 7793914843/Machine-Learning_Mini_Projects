import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    "Previous_Marks": [50, 55, 60, 65, 70, 72, 75, 80, 85, 90],
    "Final_Marks": [52, 57, 62, 67, 72, 74, 78, 83, 87, 92]
}


# 2. Convert data into DataFrame

df = pd.DataFrame(data)


# 3. Display data

print("Student Performance Data:")
print(df)


# 4. Separate input and output

X = df[["Hours_Studied", "Attendance", "Previous_Marks"]]

y = df["Final_Marks"]


# 5. Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create Linear Regression model

model = LinearRegression()


# 7. Train the model

model.fit(X_train, y_train)


# 8. Make predictions

prediction = model.predict(X_test)


# 9. Evaluate the model

mae = mean_absolute_error(y_test, prediction)

r2 = r2_score(y_test, prediction)


print("\nMean Absolute Error:", mae)

print("R2 Score:", r2)


# 10. Predict marks for a new student

new_student = pd.DataFrame([{
    "Hours_Studied": 8,
    "Attendance": 88,
    "Previous_Marks": 80
}])


result = model.predict(new_student)


print("\nPredicted Final Marks:", result[0])
