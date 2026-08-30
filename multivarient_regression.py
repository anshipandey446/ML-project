import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("FAOSTAT_data_en_8-30-2026.csv")

# Convert Element values into separate columns
data_pivot = data.pivot(
    index="Year",
    columns="Element",
    values="Value"
).reset_index()

print(data_pivot)

# Independent variables
X = data_pivot[
    [
        "Area harvested",
        "Production"
    ]
]

# Dependent variable
y = data_pivot["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMULTIPLE LINEAR REGRESSION")

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
