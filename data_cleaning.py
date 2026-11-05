import pandas as pd
from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "combined_air_quality_data.csv"
output_file = project_folder / "data" / "cleaned_hourly_air_quality.csv"

# Load combined data
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Convert pollutant columns to numeric
pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]

for column in pollutants:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# Remove duplicate timestamps
df = df.drop_duplicates(subset=["Timestamp"])

# Remove columns that contain no data
df = df.dropna(axis=1, how="all")

# Sort by timestamp
df = df.sort_values("Timestamp")

# Save cleaned data
df.to_csv(output_file, index=False)

print("\nDATA CLEANING COMPLETED")
print("Cleaned dataset shape:", df.shape)
print("\nMissing values:")
print(df[pollutants].isnull().sum())
print("\nSaved file:", output_file)