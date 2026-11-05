import joblib
import pandas as pd
from pathlib import Path

# 1. Set paths
project_folder = Path(__file__).resolve().parent
model_folder = project_folder / "models"
processed_folder = project_folder / "data" / "processed"

# 2. Load saved model and preprocessing objects
model = joblib.load(model_folder / "best_aqi_model.pkl")
imputer = joblib.load(processed_folder / "imputer.pkl")
feature_columns = joblib.load(
    processed_folder / "feature_columns.pkl"
)

# 3. Get pollutant values from user
print("\nAIR QUALITY INDEX PREDICTION SYSTEM")
print("-----------------------------------")
print("Enter the pollutant measurements:")

input_values = {}

for feature in feature_columns:
    value = float(input(f"{feature}: "))
    input_values[feature] = value

# 4. Prepare input data
input_data = pd.DataFrame(
    [input_values],
    columns=feature_columns
)

input_imputed = imputer.transform(input_data)

# 5. Predict AQI
predicted_aqi = model.predict(input_imputed)[0]
predicted_aqi = max(0, min(500, predicted_aqi))

# 6. Determine AQI category
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

category = get_category(predicted_aqi)

# 7. Display result
print("\nPREDICTION RESULT")
print("-----------------------------------")
print("Predicted AQI:", round(predicted_aqi, 2))
print("AQI Category:", category)
print("-----------------------------------")
print("This is an educational AQI estimation,")
print("not an official CPCB reading or forecast.")