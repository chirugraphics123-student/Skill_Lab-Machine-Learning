import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# Invented classroom data for learning only.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})
X = DATA[["hours_studied"]]
y = DATA["marks"]
# Start the SQLite-backed server first; see this folder's README.md.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Beginner MLflow - Regression Registry")
model_name = "StudyMarksRegressionModel"

# Alternate model type each run to create distinct registered versions if desired.
model = LinearRegression()
model.fit(X_train, y_train)
with mlflow.start_run(run_name="register-marks-regression"):
    info = mlflow.sklearn.log_model(model, name="marks_regression_model")
    version = mlflow.register_model(info.model_uri, model_name)
    print("Registered model version:", version.version)

client = MlflowClient()
client.set_registered_model_alias(model_name, "champion", version.version)
loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print("Predicted marks using champion model:", [round(float(v), 1) for v in loaded.predict(X_test.iloc[:2])])
