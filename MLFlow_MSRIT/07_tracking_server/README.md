# Lab 07 — Tracking Server Client

## Before running
1. Open the main project folder in PyCharm.
2. Select the project virtual environment as the interpreter.
3. Install dependencies from the root folder with `python -m pip install -r requirements.txt`.
4. Run this folder's Python file using the green Run button.

## Main example
This lab's example is `client_to_server.py`.

## Student submission
Read `EXERCISE.md` in this folder. It contains tasks and a checklist for the work to submit.

## MLflow UI
For labs that create runs, start the UI from the project root using:
`python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000`
Then open `http://127.0.0.1:5000` in your browser.

For this lab, start `python -m mlflow server --host 127.0.0.1 --port 5000` in one terminal before running `client_to_server.py`.
