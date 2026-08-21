import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("house.csv")

# 2. Display first 5 rows
print("First 5 rows:")
print(data.head())

# 3. Check dataset
print("\nShape:")
print(data.shape)

print("\nInformation:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# 4. Separate inputs and output
X = data.drop("Price", axis=1)
y = data["Price"]

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Create model
model = LinearRegression()

# 7. Train model
model.fit(X_train, y_train)

# 8. Make predictions
prediction = model.predict(X_test)

# 9. Evaluate model
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("\nMAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# 10. Predict a new house
new_house = [[1800, 3, 2, 4]]

new_price = model.predict(new_house)

print("\nNew House: 1800 sq ft, 3 bedrooms, 2 bathrooms, 4 years old")
print("Predicted House Price:", new_price[0])
