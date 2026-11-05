import pandas as pd
from pathlib import Path

# 1. Set file paths
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "cleaned_hourly_air_quality.csv"
output_file = project_folder / "data" / "daily_aqi_dataset.csv"

# 2. Load dataset
df = pd.read_csv(input_file)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"])
df = df.sort_values("Timestamp").copy()
df["Date"] = df["Timestamp"].dt.date

# 3. Convert pollutant columns to numeric
pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "NH3 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]

for col in pollutants:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. CPCB concentration breakpoints and AQI breakpoints
breakpoints = {
    "PM2.5 (µg/m³)": ([0, 30, 60, 90, 120, 250],
                      [0, 50, 100, 200, 300, 400, 500]),
    "PM10 (µg/m³)": ([0, 50, 100, 250, 350, 430],
                     [0, 50, 100, 200, 300, 400, 500]),
    "NO2 (µg/m³)": ([0, 40, 80, 180, 280, 400],
                    [0, 50, 100, 200, 300, 400, 500]),
    "SO2 (µg/m³)": ([0, 40, 80, 380, 800, 1600],
                    [0, 50, 100, 200, 300, 400, 500]),
    "NH3 (µg/m³)": ([0, 200, 400, 800, 1200, 1800],
                    [0, 50, 100, 200, 300, 400, 500]),
    "CO (mg/m³)": ([0, 1, 2, 10, 17, 34],
                   [0, 50, 100, 200, 300, 400, 500]),
    "Ozone (µg/m³)": ([0, 50, 100, 168, 208, 748],
                      [0, 50, 100, 200, 300, 400, 500])
}

# 5. Calculate pollutant sub-index
def calculate_sub_index(value, concentration_points, aqi_points):

    if pd.isna(value) or value < 0:
        return None

    for i in range(len(concentration_points) - 1):

        low = concentration_points[i]
        high = concentration_points[i + 1]

        if value <= high:

            aqi_low = aqi_points[i]
            aqi_high = aqi_points[i + 1]

            result = (
                (aqi_high - aqi_low) /
                (high - low)
            ) * (value - low) + aqi_low

            return round(result)

    return 500


# 6. Create daily dataset
daily = pd.DataFrame({
    "Date": sorted(df["Date"].unique())
}).set_index("Date")

# 7. Calculate 24-hour pollutant averages
twenty_four_hour_pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "NH3 (µg/m³)"
]

for col in twenty_four_hour_pollutants:

    if col in df.columns:

        daily_average = df.groupby("Date")[col].mean()
        valid_readings = df.groupby("Date")[col].count()

        daily[col] = daily_average.where(valid_readings >= 16)

# 8. Calculate 8-hour averages for CO and Ozone
for col in ["CO (mg/m³)", "Ozone (µg/m³)"]:

    if col in df.columns:

        rolling_average = df.groupby("Date")[col].transform(
            lambda values: values.rolling(
                window=8,
                min_periods=8
            ).mean()
        )

        temp = pd.DataFrame({
            "Date": df["Date"],
            "Rolling_Average": rolling_average
        })

        maximum_8_hour_average = temp.groupby("Date")[
            "Rolling_Average"
        ].max()

        valid_readings = df.groupby("Date")[col].count()

        daily[col] = maximum_8_hour_average.where(
            valid_readings >= 16
        )

# 9. Calculate individual pollutant sub-indices
for col, (concentration_points, aqi_points) in breakpoints.items():

    if col in daily.columns:

        daily[col + "_sub_index"] = daily[col].apply(
            lambda value: calculate_sub_index(
                value,
                concentration_points,
                aqi_points
            )
        )

# 10. Calculate overall AQI
sub_index_columns = [
    col + "_sub_index"
    for col in breakpoints
    if col + "_sub_index" in daily.columns
]

def calculate_overall_aqi(row):

    valid_indices = row[sub_index_columns].dropna()

    pm_available = (
        pd.notna(row.get("PM2.5 (µg/m³)_sub_index")) or
        pd.notna(row.get("PM10 (µg/m³)_sub_index"))
    )

    if len(valid_indices) >= 3 and pm_available:
        return int(valid_indices.max())

    return None

daily["AQI"] = daily.apply(calculate_overall_aqi, axis=1)

# 11. Add AQI category
def get_category(aqi):

    if pd.isna(aqi):
        return "Insufficient Data"
    elif aqi <= 50:
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

daily["AQI_Category"] = daily["AQI"].apply(get_category)

# 12. Save final dataset
daily = daily.reset_index()

daily.to_csv(output_file, index=False)

# 13. Display results
print("\nAQI CALCULATION COMPLETED")
print("--------------------------------")

print("Daily dataset shape:", daily.shape)

print("\nDays with calculated AQI:")
print(daily["AQI"].notna().sum())

print("\nDays with insufficient data:")
print(daily["AQI"].isna().sum())

print("\nAQI category distribution:")
print(daily["AQI_Category"].value_counts())

print("\nFirst five rows:")
print(daily.head())

print("\nSaved successfully at:")
print(output_file)