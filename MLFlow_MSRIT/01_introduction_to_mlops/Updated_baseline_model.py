import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Invented classroom data for learning only; not real student-performance research.
DATA = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
    "marks": [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92],
})

X = DATA[["hours_studied"]]
y = DATA["marks"]

# CHANGED: Updated test_size from 0.25 to 0.30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual marks: ", y_test.tolist())
print("Predicted marks:", [round(float(value), 1) for value in predictions])
print("Mean Absolute Error (lower is better):", round(mean_absolute_error(y_test, predictions), 2))
print("R-squared (closer to 1 is generally better):", round(r2_score(y_test, predictions), 2))
print("No MLflow used yet: results are only printed, not tracked.")

'''• Why predicting marks is a regression task: Predicting student marks is a 
regression task because the target output is a continuous numeric value that 
can fall anywhere along an infinite numerical range, rather than a discrete 
class or category label.
• 
Importance of saving code and evaluation results (MLOps context): 
Saving both the exact model code and its performance metrics 
is vital because it establishes a reliable baseline for reproducibility 
and version control. If a future model modification causes a drop in 
performance or unpredictable errors, engineers need to trace back to 
previous versions to see exactly what changed in the source file and
compare results side-by-side. This documentation avoids guesswork and ensures code updates genuinely improve accuracy instead of causing performance regressions'''