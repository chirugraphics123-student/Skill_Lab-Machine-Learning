import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# 1. DATASET
# --------------------------------------------------
data = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 8, 3, 5, 7, 9, 1, 5, 7, 9
    ],
    "marks": [
        35, 40, 48, 52, 60, 65, 72, 80, 42, 55,
        63, 78, 46, 58, 75, 88, 32, 62, 70, 92
    ]
})

# --------------------------------------------------
# 2. EXPERIMENT CONFIGURATION
# --------------------------------------------------
RANDOM_SEED = 7
TEST_SIZE = 0.25

FEATURES = ["hours_studied"]
TARGET = "marks"

# --------------------------------------------------
# 3. PREPARE TRAINING AND TEST DATA
# --------------------------------------------------
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

# --------------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. EVALUATE THE MODEL
# --------------------------------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nEXPERIMENT RESULTS")
print("------------------")
print("Random seed:", RANDOM_SEED)
print("Test size:", TEST_SIZE)
print("Features:", FEATURES)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Model intercept:", round(model.intercept_, 4))
print("Model coefficient(s):", model.coef_)
print("MAE:", round(mae, 4))
print("R-squared:", round(r2, 4))