"""
First ML example: Predict marks from study hours.

Classroom goals:
1. Split data into training and test sets.
2. Calculate linear regression slope and intercept by hand.
3. Predict marks for test observations.
4. Calculate MAE and R-squared by hand.
5. Compare manual calculations with scikit-learn.

Note:
The data is invented for teaching and is not real student-performance research.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ------------------------------------------------------------
# 1. Prepare the classroom dataset
# ------------------------------------------------------------
DATA = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 8, 3, 5, 7, 9, 1, 5, 7, 9
    ],
    "marks": [
        35, 40, 48, 52, 60, 65, 72, 80, 42, 55,
        63, 78, 46, 58, 75, 88, 32, 62, 70, 92
    ],
})

print("=" * 72)
print("LINEAR REGRESSION: PREDICT MARKS FROM STUDY HOURS")
print("=" * 72)
print("\nFull classroom dataset:")
print(DATA.to_string(index=False))


# ------------------------------------------------------------
# 2. Split data into training and test sets
# ------------------------------------------------------------
X = DATA[["hours_studied"]]
y = DATA["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# Make a table that is easier to inspect in class.
train_data = pd.DataFrame({
    "hours_studied": X_train["hours_studied"],
    "marks": y_train,
}).sort_index()

test_data = pd.DataFrame({
    "hours_studied": X_test["hours_studied"],
    "marks": y_test,
}).sort_index()

print("\n" + "-" * 72)
print("STEP 1: TRAIN / TEST SPLIT")
print("-" * 72)
print(f"Total observations: {len(DATA)}")
print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")
print("\nTraining observations (original row order):")
print(train_data.to_string(index=False))
print("\nTest observations (original row order):")
print(test_data.to_string(index=False))


# ------------------------------------------------------------
# 3. Calculate slope and intercept by hand
# ------------------------------------------------------------
# For y_hat = m*x + c:
#
# m = [n*sum(x*y) - sum(x)*sum(y)] /
#     [n*sum(x^2) - (sum(x))^2]
#
# c = [sum(y) - m*sum(x)] / n
#
# Use training data only. Do not use test data to fit the model.

x_values = X_train["hours_studied"].astype(float)
y_values = y_train.astype(float)

n = len(x_values)
sum_x = x_values.sum()
sum_y = y_values.sum()
sum_x_squared = (x_values ** 2).sum()
sum_xy = (x_values * y_values).sum()

numerator = n * sum_xy - sum_x * sum_y
denominator = n * sum_x_squared - sum_x ** 2

if denominator == 0:
    raise ValueError("Cannot calculate slope: all training X values are identical.")

m_manual = numerator / denominator
c_manual = (sum_y - m_manual * sum_x) / n

calculation_table = pd.DataFrame({
    "x (hours)": x_values,
    "y (marks)": y_values,
    "x^2": x_values ** 2,
    "x*y": x_values * y_values,
}).sort_index()

print("\n" + "-" * 72)
print("STEP 2: MANUAL CALCULATION OF SLOPE AND INTERCEPT")
print("-" * 72)
print("\nTraining-data calculation table:")
print(calculation_table.to_string(index=False, formatters={
    "x (hours)": "{:.0f}".format,
    "y (marks)": "{:.0f}".format,
    "x^2": "{:.0f}".format,
    "x*y": "{:.0f}".format,
}))

print("\nRequired totals from training data:")
print(f"n             = {n}")
print(f"sum(x)        = {sum_x:.0f}")
print(f"sum(y)        = {sum_y:.0f}")
print(f"sum(x^2)      = {sum_x_squared:.0f}")
print(f"sum(x*y)      = {sum_xy:.0f}")

print("\nSlope formula:")
print("m = [n*sum(x*y) - sum(x)*sum(y)] / [n*sum(x^2) - (sum(x))^2]")
print(f"m = [{n}*{sum_xy:.0f} - ({sum_x:.0f}*{sum_y:.0f})] / "
      f"[{n}*{sum_x_squared:.0f} - ({sum_x:.0f})^2]")
print(f"m = {numerator:.0f} / {denominator:.0f}")
print(f"m = {m_manual:.6f}")

print("\nIntercept formula:")
print("c = [sum(y) - m*sum(x)] / n")
print(f"c = [{sum_y:.0f} - ({m_manual:.6f}*{sum_x:.0f})] / {n}")
print(f"c = {c_manual:.6f}")

print("\nLearned equation from manual calculations:")
print(f"predicted_marks = {m_manual:.6f} * hours_studied + {c_manual:.6f}")


# ------------------------------------------------------------
# 4. Predict test marks using the manual equation
# ------------------------------------------------------------
manual_predictions = m_manual * X_test["hours_studied"].astype(float) + c_manual

prediction_table = pd.DataFrame({
    "hours_studied": X_test["hours_studied"],
    "actual_marks": y_test,
    "manual_prediction": manual_predictions,
}).sort_index()

prediction_table["error (actual - predicted)"] = (
    prediction_table["actual_marks"] - prediction_table["manual_prediction"]
)
prediction_table["absolute_error"] = prediction_table[
    "error (actual - predicted)"
].abs()
prediction_table["squared_error"] = prediction_table[
    "error (actual - predicted)"
] ** 2

print("\n" + "-" * 72)
print("STEP 3: PREDICT THE TEST OBSERVATIONS")
print("-" * 72)
print(prediction_table.to_string(index=False, formatters={
    "hours_studied": "{:.0f}".format,
    "actual_marks": "{:.0f}".format,
    "manual_prediction": "{:.4f}".format,
    "error (actual - predicted)": "{:.4f}".format,
    "absolute_error": "{:.4f}".format,
    "squared_error": "{:.4f}".format,
}))


# ------------------------------------------------------------
# 5. Calculate MAE by hand
# ------------------------------------------------------------
mae_manual = prediction_table["absolute_error"].mean()

print("\n" + "-" * 72)
print("STEP 4: MEAN ABSOLUTE ERROR (MAE)")
print("-" * 72)
print("MAE = sum(abs(actual - predicted)) / number_of_test_observations")
print(f"Sum of absolute errors = "
      f"{prediction_table['absolute_error'].sum():.6f}")
print(f"Number of test observations = {len(prediction_table)}")
print(f"Manual MAE = {mae_manual:.6f}")


# ------------------------------------------------------------
# 6. Calculate R-squared by hand
# ------------------------------------------------------------
# R^2 = 1 - SS_res / SS_tot
# SS_res = sum((actual - predicted)^2)
# SS_tot = sum((actual - mean(actual test marks))^2)

test_mean = y_test.mean()
ss_res = prediction_table["squared_error"].sum()
ss_tot = ((y_test.astype(float) - test_mean) ** 2).sum()

if ss_tot == 0:
    r2_manual = float("nan")
else:
    r2_manual = 1 - ss_res / ss_tot

print("\n" + "-" * 72)
print("STEP 5: R-SQUARED BY HAND")
print("-" * 72)
print(f"Mean of actual test marks = {test_mean:.6f}")
print(f"Residual sum of squares (SS_res) = {ss_res:.6f}")
print(f"Total sum of squares (SS_tot) = {ss_tot:.6f}")
print("R^2 = 1 - SS_res / SS_tot")
print(f"R^2 = 1 - ({ss_res:.6f} / {ss_tot:.6f})")
print(f"Manual R-squared = {r2_manual:.6f}")


# ------------------------------------------------------------
# 7. Verify calculations with scikit-learn
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

sklearn_predictions = model.predict(X_test)
mae_sklearn = mean_absolute_error(y_test, sklearn_predictions)
r2_sklearn = r2_score(y_test, sklearn_predictions)

print("\n" + "-" * 72)
print("STEP 6: VERIFY WITH SCIKIT-LEARN")
print("-" * 72)
print(f"Manual slope:       {m_manual:.6f}")
print(f"Scikit-learn slope: {model.coef_[0]:.6f}")
print(f"Manual intercept:       {c_manual:.6f}")
print(f"Scikit-learn intercept: {model.intercept_:.6f}")
print(f"Manual MAE:        {mae_manual:.6f}")
print(f"Scikit-learn MAE:  {mae_sklearn:.6f}")
print(f"Manual R-squared:       {r2_manual:.6f}")
print(f"Scikit-learn R-squared: {r2_sklearn:.6f}")

print("\nScikit-learn predictions in the test-set order:")
for actual, predicted in zip(y_test.tolist(), sklearn_predictions):
    print(f"Actual = {actual:>5.1f}, predicted = {predicted:>8.4f}")

# Use assertions to check that the hand calculations match sklearn.
assert abs(m_manual - model.coef_[0]) < 1e-9
assert abs(c_manual - model.intercept_) < 1e-9
assert abs(mae_manual - mae_sklearn) < 1e-9
assert abs(r2_manual - r2_sklearn) < 1e-9

print("\nVerification successful: manual calculations match scikit-learn.")
print("No MLflow used yet: results are printed but not tracked.")
