import mlflow
from pathlib import Path

mlflow.set_experiment("Beginner MLflow - Tracking")
with mlflow.start_run(run_name="tracking-basics") as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae_demo", 4.2)
    Path("my_note.txt").write_text(
        "Today I learned that MLflow records experiment runs. In regression, MAE measures average prediction error.\n",
        encoding="utf-8"
    )
    mlflow.log_artifact("my_note.txt")
    print("Experiment ID:", run.info.experiment_id)
    print("Run ID:", run.info.run_id)
    print("A text file was logged as an artifact.")
