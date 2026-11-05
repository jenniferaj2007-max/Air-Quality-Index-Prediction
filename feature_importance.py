import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent
model_folder = project_folder / "models"
processed_folder = project_folder / "data" / "processed"
visualization_folder = project_folder / "visualizations"

visualization_folder.mkdir(parents=True, exist_ok=True)

# Load Random Forest model and feature names
model = joblib.load(model_folder / "best_aqi_model.pkl")
features = joblib.load(
    processed_folder / "feature_columns.pkl"
)

# Extract feature importance
importance = pd.DataFrame({
    "Pollutant": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nPOLLUTANT FEATURE IMPORTANCE")
print("--------------------------------")
print(importance.to_string(index=False))

# Create chart
plt.figure(figsize=(9, 5))
plt.barh(importance["Pollutant"], importance["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Pollutants")
plt.title("Random Forest - Pollutant Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    visualization_folder / "feature_importance.png"
)
plt.show()
plt.close()

print("\nFeature importance chart saved successfully!")