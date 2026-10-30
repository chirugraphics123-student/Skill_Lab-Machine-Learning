from mlflow.tracking import MlflowClient

client = MlflowClient()
print("Experiments saved in this tracking store:")
for experiment in client.search_experiments():
    print(f"- {experiment.name} (ID: {experiment.experiment_id})")

experiment = client.get_experiment_by_name("Beginner MLflow - Compare Regression Runs")
if experiment is None:
    print("\nNo comparison runs found yet. Run folder 05 first.")
else:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=10
    )
    print("\nRuns sorted by MAE (lowest error first):")
    for run in runs:
        print(
            "run_id =", run.info.run_id,
            "| MAE =", run.data.metrics.get("mae"),
            "| params =", run.data.params
        )
