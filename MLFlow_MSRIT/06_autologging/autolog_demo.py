import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Invented classroom data for learning only.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})
X = DATA[["hours_studied"]]
y = DATA["marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Beginner MLflow - Regression Autologging")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="automatic-regression-logging"):
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=7)
    model.fit(X_train, y_train)
    print("Model trained. Check the parameters, model, and available metrics in the MLflow UI.")

mlflow.sklearn.autolog(disable=True)
