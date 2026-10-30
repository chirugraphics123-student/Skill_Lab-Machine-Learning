import mlflow
from pathlib import Path

mlflow.set_experiment("Beginner MLflow - Tracking")
with mlflow.start_run(run_name="tracking-basics") as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae_demo", 4.2)
    Path("my_note.txt").write_text(
        "Today I learned that MLflow tracks experiments by logging parameters, metrics, and artifacts. This makes it easy to reproduce results and compare different runs.\n",
        encoding="utf-8"
    )
    mlflow.log_artifact("my_note.txt")


    Path("next_steps.txt").write_text(
        "Next steps: explore multi-run comparisons, learn about model registry, and try logging multiple metrics.\n",
        encoding="utf-8"
    )
    mlflow.log_artifact("next_steps.txt")

    print("Experiment ID:", run.info.experiment_id)
    print("Run ID:", run.info.run_id)
    print("Both artifact files have been logged successfully.")