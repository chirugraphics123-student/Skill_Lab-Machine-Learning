import pandas as pd
import mlflow
from sklearn.linear_model import Ridge
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
mlflow.set_experiment("Beginner MLflow - Compare Regression Runs")

# Ridge is a linear regression model with a tunable regularization strength (alpha).
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    with mlflow.start_run(run_name=f"ridge-alpha-{alpha}"):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        print(f"alpha={alpha}, MAE={mae:.2f}, R-squared={r2:.2f}")
print("Open the MLflow UI to compare the three runs. Lower MAE is better.")
