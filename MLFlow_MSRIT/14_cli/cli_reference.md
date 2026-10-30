# Lab 14 — MLflow Command-Line Interface (CLI)

## Aim
Practise the MLflow commands used to inspect the CLI, open the UI, run a project, and learn about model serving.

## Before you begin
Open the PyCharm Terminal in the project root and make sure the virtual environment is active.

## Commands to try

```bash
python -m mlflow --version
python -m mlflow --help
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser while the UI command is running. Stop it with Ctrl+C.

From the `12_projects` directory, try:

```bash
python -m mlflow run . -P alpha=0.1
python -m mlflow run . -P alpha=10.0
```

To learn about additional commands without running them yet:

```bash
python -m mlflow artifacts download --help
python -m mlflow models serve --help
python -m mlflow server --help
```

Use `127.0.0.1` for classroom exercises. Do not expose a server to the public internet.
