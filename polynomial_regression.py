import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("FAOSTAT_data_en_8-30-2026.csv")

X = data[["Year"]]
y = data["Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = LinearRegression()

model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

print("POLYNOMIAL REGRESSION")

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

X_sorted = np.sort(X["Year"].values).reshape(-1, 1)

X_sorted_poly = poly.transform(
    pd.DataFrame(X_sorted, columns=["Year"])
)

y_sorted_pred = model.predict(X_sorted_poly)

plt.scatter(X, y)
plt.plot(X_sorted, y_sorted_pred)

plt.xlabel("Year")
plt.ylabel("Wheat Yield (kg/ha)")
plt.title("Polynomial Regression: Year vs Wheat Yield")
plt.show()
