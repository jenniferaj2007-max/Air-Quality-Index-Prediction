import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

project_folder = Path(__file__).resolve().parent
processed_folder = project_folder / "data" / "processed"
output_folder = project_folder / "outputs"
output_folder.mkdir(parents=True, exist_ok=True)

y_test = pd.read_csv(
    processed_folder / "y_test.csv"
).squeeze()

linear = pd.read_csv(
    output_folder / "linear_regression_predictions.csv"
)

forest = pd.read_csv(
    output_folder / "random_forest_predictions.csv"
)

models = {
    "Linear Regression": linear["Predicted AQI"],
    "Random Forest": forest["Predicted AQI"]
}

results = []

for name, predictions in models.items():
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

comparison = pd.DataFrame(results)

comparison.to_csv(
    output_folder / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("-----------------------------")
print(comparison.round(4).to_string(index=False))

best_model = comparison.loc[
    comparison["MAE"].idxmin(), "Model"
]

print("\nBest model based on MAE:", best_model)
print("\nComparison saved successfully!")