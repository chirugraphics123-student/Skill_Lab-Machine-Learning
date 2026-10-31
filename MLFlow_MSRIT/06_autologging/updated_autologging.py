import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Invented classroom data for learning only.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks": [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})

X = DATA[["hours_studied"]]
y = DATA["marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Beginner MLflow - Regression Autologging")
mlflow.sklearn.autolog()

# Run 1: more trees, deeper
with mlflow.start_run(run_name="50-trees-depth3"):
    model1 = RandomForestRegressor(n_estimators=50, max_depth=3)
    model1.fit(X_train, y_train)
    mae1 = mean_absolute_error(y_test, model1.predict(X_test))
    mlflow.log_metric("test_mae", mae1)
    print(f"Run 1 → n_estimators=50, max_depth=3, test_mae={mae1:.2f}")

# Run 2: fewer trees, shallower
with mlflow.start_run(run_name="10-trees-depth2"):
    model2 = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=7)
    model2.fit(X_train, y_train)
    mae2 = mean_absolute_error(y_test, model2.predict(X_test))
    mlflow.log_metric("test_mae", mae2)
    print(f"Run 2 → n_estimators=10, max_depth=2, test_mae={mae2:.2f}")

mlflow.sklearn.autolog(disable=True)
print("Open http://localhost:5000 to compare both runs.")