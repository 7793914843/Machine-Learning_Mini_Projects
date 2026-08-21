import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("house.csv")

# Display data
print("First 5 rows:")
print(data.head())

# Check information
print("\nShape:")
print(data.shape)

print("\nInformation:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# Separate X and y
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("\nMAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# New house
new_house = [[1800, 3, 2, 4]]

new_price = model.predict(new_house)

print("\nPredicted House Price:", new_price[0])
