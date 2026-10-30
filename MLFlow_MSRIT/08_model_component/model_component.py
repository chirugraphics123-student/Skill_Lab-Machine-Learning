import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Invented classroom data for learning only.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})
X = DATA[["hours_studied"]]
y = DATA["marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = LinearRegression()
model.fit(X_train, y_train)
signature = infer_signature(X_train, model.predict(X_train))

mlflow.set_experiment("Beginner MLflow - Regression Model")
with mlflow.start_run(run_name="save-and-load-regression-model"):
    info = mlflow.sklearn.log_model(
        model, name="marks_regression_model",
        signature=signature, input_example=X_train.iloc[[0]]
    )
    print("Saved model URI:", info.model_uri)

loaded_model = mlflow.sklearn.load_model(info.model_uri)
example = X_test.iloc[[0, 1]]
print("Hours studied:", example["hours_studied"].tolist())
print("Original model predictions:", [round(float(v), 1) for v in model.predict(example)])
print("Loaded model predictions:  ", [round(float(v), 1) for v in loaded_model.predict(example)])
