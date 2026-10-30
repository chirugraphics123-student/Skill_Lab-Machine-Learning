import mlflow
from pathlib import Path

mlflow.set_experiment("Beginner MLflow - Logging")
with mlflow.start_run(run_name="logging-basics"):
    mlflow.log_param("algorithm", "linear_regression")
    mlflow.log_params({"test_size": 0.25, "random_seed": 7})
    mlflow.log_metric("mae", 4.2)
    mlflow.log_metrics({"r2": 0.91, "number_of_features": 1})
    mlflow.set_tag("class", "first-year")
    mlflow.set_tag("purpose", "regression-practice")
    Path("lab_note.txt").write_text("My first logged regression note.", encoding="utf-8")
    mlflow.log_artifact("lab_note.txt")
    print("Logged parameters, regression metrics, tags, and one artifact.")
