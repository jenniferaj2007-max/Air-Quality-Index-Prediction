import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Set paths
project_folder = Path(__file__).resolve().parent
processed_folder = project_folder / "data" / "processed"
model_folder = project_folder / "models"
output_folder = project_folder / "outputs"

model_folder.mkdir(parents=True, exist_ok=True)
output_folder.mkdir(parents=True, exist_ok=True)

# 2. Load processed data
X_train = pd.read_csv(processed_folder / "X_train.csv")
X_test = pd.read_csv(processed_folder / "X_test.csv")

y_train = pd.read_csv(processed_folder / "y_train.csv").squeeze()
y_test = pd.read_csv(processed_folder / "y_test.csv").squeeze()

# 3. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# 4. Train model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRANDOM FOREST REGRESSION RESULTS")
print("--------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 7. Save model
joblib.dump(
    model,
    model_folder / "random_forest_model.pkl"
)

# 8. Save predictions
results = pd.DataFrame({
    "Actual AQI": y_test,
    "Predicted AQI": y_pred
})

results.to_csv(
    output_folder / "random_forest_predictions.csv",
    index=False
)

print("\nRandom Forest model saved successfully!")