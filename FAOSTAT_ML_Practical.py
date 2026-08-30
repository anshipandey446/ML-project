import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. LOAD FAOSTAT DATA
# ============================================================

data = pd.read_csv("FAOSTAT_data_en_8-30-2026.csv")

print("================================================")
print("FAOSTAT DATA")
print("================================================")

print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)


# ============================================================
# 2. CHECK COUNTRY, CROP AND ELEMENT
# ============================================================

print("\nCountry:")
print(data["Area"].unique())

print("\nCrop:")
print(data["Item"].unique())

print("\nElement:")
print(data["Element"].unique())


# ============================================================
# 3. YEAR AND YIELD DATA
# ============================================================

print("\nYear and Yield:")
print(data[["Year", "Value"]])


# ============================================================
# 4. SIMPLE LINEAR REGRESSION
# Year -> Wheat Yield
# ============================================================

X = data[["Year"]]
y = data["Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)


# ============================================================
# 5. LINEAR REGRESSION RESULTS
# ============================================================

print("\n================================================")
print("LINEAR REGRESSION")
print("================================================")

print("Coefficient:", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred_linear)

linear_mse = mean_squared_error(
    y_test,
    y_pred_linear
)

linear_r2 = r2_score(
    y_test,
    y_pred_linear
)

print("\nMean Squared Error:", linear_mse)
print("R2 Score:", linear_r2)


# ============================================================
# 6. LINEAR REGRESSION GRAPH
# ============================================================

plt.figure()

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    linear_model.predict(X),
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Year")
plt.ylabel("Wheat Yield (kg/ha)")
plt.title("Linear Regression: Year vs Wheat Yield")
plt.legend()

plt.show()


# ============================================================
# 7. POLYNOMIAL REGRESSION
# Year -> Wheat Yield
# ============================================================

poly = PolynomialFeatures(degree=2)

X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(
    X_poly_train,
    y_train
)

y_pred_poly = poly_model.predict(
    X_poly_test
)


# ============================================================
# 8. POLYNOMIAL REGRESSION RESULTS
# ============================================================

print("\n================================================")
print("POLYNOMIAL REGRESSION")
print("================================================")

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred_poly)

poly_mse = mean_squared_error(
    y_test,
    y_pred_poly
)

poly_r2 = r2_score(
    y_test,
    y_pred_poly
)

print("\nMean Squared Error:", poly_mse)
print("R2 Score:", poly_r2)


# ============================================================
# 9. POLYNOMIAL REGRESSION GRAPH
# ============================================================

X_sorted = np.sort(
    X.values,
    axis=0
)

X_sorted_poly = poly.transform(
    X_sorted
)

y_sorted_pred = poly_model.predict(
    X_sorted_poly
)

plt.figure()

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_sorted,
    y_sorted_pred,
    linewidth=2,
    label="Polynomial Curve"
)

plt.xlabel("Year")
plt.ylabel("Wheat Yield (kg/ha)")
plt.title("Polynomial Regression: Year vs Wheat Yield")
plt.legend()

plt.show()


# ============================================================
# 10. LOAD MULTIPLE REGRESSION DATA
# ============================================================

multi_data = pd.read_csv("wheat_multiple.csv")

print("\n================================================")
print("MULTIPLE REGRESSION DATA")
print("================================================")

print(multi_data.head())

print("\nColumns:")
print(multi_data.columns)


# ============================================================
# 11. CONVERT ELEMENTS INTO SEPARATE COLUMNS
# ============================================================

multi_pivot = multi_data.pivot(
    index="Year",
    columns="Element",
    values="Value"
).reset_index()

print("\n================================================")
print("PIVOTED DATA")
print("================================================")

print(multi_pivot)


# ============================================================
# 12. MULTIPLE REGRESSION VARIABLES
# Area harvested + Production -> Yield
# ============================================================

X_multi = multi_pivot[
    [
        "Area harvested",
        "Production"
    ]
]

y_multi = multi_pivot["Yield"]

print("\nIndependent variables:")
print(X_multi)

print("\nDependent variable:")
print(y_multi)


# ============================================================
# 13. TRAIN-TEST SPLIT
# ============================================================

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 14. MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

multi_model = LinearRegression()

multi_model.fit(
    X_train_multi,
    y_train_multi
)

y_pred_multi = multi_model.predict(
    X_test_multi
)


# ============================================================
# 15. MULTIPLE REGRESSION RESULTS
# ============================================================

print("\n================================================")
print("MULTIPLE LINEAR REGRESSION")
print("================================================")

print("Coefficients:")
print(multi_model.coef_)

print("\nIntercept:")
print(multi_model.intercept_)

print("\nActual values:")
print(y_test_multi.values)

print("\nPredicted values:")
print(y_pred_multi)


# ============================================================
# 16. MULTIPLE REGRESSION EVALUATION
# ============================================================

multi_mse = mean_squared_error(
    y_test_multi,
    y_pred_multi
)

multi_r2 = r2_score(
    y_test_multi,
    y_pred_multi
)

print("\nMean Squared Error:", multi_mse)
print("R2 Score:", multi_r2)


# ============================================================
# 17. FINAL MODEL COMPARISON
# ============================================================

print("\n================================================")
print("FINAL MODEL COMPARISON")
print("================================================")

print("\nLinear Regression")
print("MSE:", linear_mse)
print("R2:", linear_r2)

print("\nPolynomial Regression")
print("MSE:", poly_mse)
print("R2:", poly_r2)

print("\nMultiple Linear Regression")
print("MSE:", multi_mse)
print("R2:", multi_r2)

print("\n================================================")
print("PRACTICAL COMPLETED")
print("================================================")
