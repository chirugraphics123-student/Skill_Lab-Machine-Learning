import os
import mlflow

# Start the server in a separate terminal before running this script.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Beginner MLflow - Tracking Server")

with mlflow.start_run(run_name="student-regression-client"):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("demo_mae", 4.2)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Regression demo run sent to the tracking server.")
