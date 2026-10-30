import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel

class SimpleMarksModel(PythonModel):
    """Illustrative custom pyfunc model: estimate marks from study hours."""
    def predict(self, context, model_input):
        data = pd.DataFrame(model_input)
        # Simple hand-written rule for demonstrating a custom model interface.
        predicted_marks = 30 + 6.5 * data["hours_studied"]
        return pd.DataFrame({"predicted_marks": predicted_marks.clip(0, 100).round(1)})

example = pd.DataFrame({"hours_studied": [1, 4, 7]})
mlflow.set_experiment("Beginner MLflow - Custom Regression Model")
with mlflow.start_run(run_name="custom-marks-model"):
    info = mlflow.pyfunc.log_model(
        name="custom_marks_model",
        python_model=SimpleMarksModel(),
        input_example=example
    )
    print("Model URI:", info.model_uri)

loaded = mlflow.pyfunc.load_model(info.model_uri)
print(loaded.predict(example))
