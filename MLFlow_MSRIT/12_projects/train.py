import argparse
import mlflow
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()

data = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})
X, y = data[["hours_studied"]], data["marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
mlflow.set_experiment("Beginner MLflow - Regression Projects")
model = Ridge(alpha=args.alpha)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
with mlflow.start_run():
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
print("alpha:", args.alpha)
print("MAE:", round(mae, 2))
print("R-squared:", round(r2, 2))
