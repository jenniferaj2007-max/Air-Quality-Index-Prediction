import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
file_path = project_folder / "data" / "combined_air_quality_data.csv"

df = pd.read_csv(file_path)

print("DATASET INFORMATION")
print("-------------------")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nStatistical Summary:")
print(df.describe(include="all"))
print("\nACTUAL COLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET SIZE:")
print(df.shape)

print("\nTIMESTAMP SAMPLE:")
print(df["Timestamp"].head(10))

print("\nPOLLUTANT DATA AVAILABILITY:")
print(df[[
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]].notna().sum())

# Check daily data availability

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]

daily_counts = (
    df.set_index("Timestamp")[pollutants]
    .resample("D")
    .count()
)

pm_available = (
    daily_counts[["PM2.5 (µg/m³)", "PM10 (µg/m³)"]]
    .ge(16).any(axis=1)
)

three_pollutants = daily_counts.ge(16).sum(axis=1) >= 3

valid_days = pm_available & three_pollutants

print("\nDATE RANGE:")
print(df["Timestamp"].min(), "to", df["Timestamp"].max())

print("\nTOTAL DAYS:", len(daily_counts))
print("DAYS WITH SUFFICIENT POLLUTANT DATA:", valid_days.sum())