import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Set paths
project_folder = Path(__file__).resolve().parent
processed_folder = project_folder / "data" / "processed"
model_folder = project_folder / "models"

model_folder.mkdir(parents=True, exist_ok=True)

# 2. Load processed data
X_train = pd.read_csv(processed_folder / "X_train_scaled.csv")
X_test = pd.read_csv(processed_folder / "X_test_scaled.csv")

y_train = pd.read_csv(processed_folder / "y_train.csv").squeeze()
y_test = pd.read_csv(processed_folder / "y_test.csv").squeeze()

# 3. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("--------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 6. Save model
joblib.dump(
    model,
    model_folder / "linear_regression_model.pkl"
)

# 7. Save predictions
results = pd.DataFrame({
    "Actual AQI": y_test,
    "Predicted AQI": y_pred
})

results.to_csv(
    project_folder / "outputs" / "linear_regression_predictions.csv",
    index=False
)

print("\nLinear Regression model saved successfully!")