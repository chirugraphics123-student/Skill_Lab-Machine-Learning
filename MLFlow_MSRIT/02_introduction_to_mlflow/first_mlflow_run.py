import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Invented classroom data for learning only.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})
X = DATA[["hours_studied"]]
y = DATA["marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = LinearRegression()



with mlflow.start_run(run_name="my-first-regression-run") as run:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.25)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, name="marks_regression_model")
    print("Run ID:", run.info.run_id)
    print("MAE:", round(mae, 2))
    print("R-squared:", round(r2, 2))
