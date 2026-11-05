import pandas as pd
import joblib
import shutil
from pathlib import Path

# 1. Set paths
project_folder = Path(__file__).resolve().parent
comparison_file = project_folder / "outputs" / "model_comparison.csv"
model_folder = project_folder / "models"

# 2. Load comparison results
comparison = pd.read_csv(comparison_file)

# 3. Select model with the lowest MAE
best_row = comparison.loc[comparison["MAE"].idxmin()]
best_model_name = best_row["Model"]

# 4. Identify the saved model file
model_files = {
    "Linear Regression": "linear_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

source_file = model_folder / model_files[best_model_name]
best_model_file = model_folder / "best_aqi_model.pkl"

# 5. Copy the best model
shutil.copy2(source_file, best_model_file)

# 6. Load and verify saved model
best_model = joblib.load(best_model_file)

print("\nBEST MODEL SELECTION COMPLETED")
print("--------------------------------")
print("Selected model:", best_model_name)
print("MAE:", round(best_row["MAE"], 4))
print("RMSE:", round(best_row["RMSE"], 4))
print("R2 Score:", round(best_row["R2 Score"], 4))
print("\nBest model saved as:", best_model_file.name)