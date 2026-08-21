import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
data = pd.read_csv("sales_data.csv")

print("First 5 rows:")
print(data.head())


# 2. Select input features
X = data[[
    "Advertising",
    "Price",
    "Discount",
    "PreviousSales"
]]


# 3. Select target
y = data["Sales"]


# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create Linear Regression model
model = LinearRegression()


# 6. Train the model
model.fit(X_train, y_train)


# 7. Make predictions
prediction = model.predict(X_test)


# 8. Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, prediction)

print("\nMean Absolute Error:", mae)


# 9. Calculate Mean Squared Error
mse = mean_squared_error(y_test, prediction)

print("Mean Squared Error:", mse)


# 10. Calculate R2 Score
r2 = r2_score(y_test, prediction)

print("R2 Score:", r2)


# 11. Predict sales for new data
new_data = [[
    25000,     # Advertising
    420,       # Price
    18,        # Discount
    200        # Previous Sales
]]

predicted_sales = model.predict(new_data)

print("\nPredicted Sales:", predicted_sales[0])
