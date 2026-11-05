import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Set paths
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "daily_aqi_dataset.csv"
output_folder = project_folder / "data" / "processed"

output_folder.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(input_file)

# 3. Select input features and target
feature_columns = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "NH3 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]

feature_columns = [
    col for col in feature_columns if col in df.columns
]

df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

df = df.dropna(subset=["AQI"])

X = df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)
y = df["AQI"]

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 5. Handle missing values using training data only
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 6. Scale features for Linear Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 7. Save processed datasets
pd.DataFrame(
    X_train_imputed, columns=feature_columns
).to_csv(output_folder / "X_train.csv", index=False)

pd.DataFrame(
    X_test_imputed, columns=feature_columns
).to_csv(output_folder / "X_test.csv", index=False)

pd.Series(y_train).to_csv(
    output_folder / "y_train.csv", index=False, header=["AQI"]
)

pd.Series(y_test).to_csv(
    output_folder / "y_test.csv", index=False, header=["AQI"]
)

pd.DataFrame(
    X_train_scaled, columns=feature_columns
).to_csv(output_folder / "X_train_scaled.csv", index=False)

pd.DataFrame(
    X_test_scaled, columns=feature_columns
).to_csv(output_folder / "X_test_scaled.csv", index=False)

# 8. Save preprocessing objects
joblib.dump(imputer, output_folder / "imputer.pkl")
joblib.dump(scaler, output_folder / "scaler.pkl")
joblib.dump(feature_columns, output_folder / "feature_columns.pkl")

# 9. Display results
print("\nDATA PREPROCESSING COMPLETED")
print("--------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", len(feature_columns))
print("\nFeatures:", feature_columns)
print("\nTraining data shape:", X_train_imputed.shape)
print("Testing data shape:", X_test_imputed.shape)
print("\nProcessed files saved in:", output_folder)