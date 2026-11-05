import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Set project paths
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "daily_aqi_dataset.csv"
output_folder = project_folder / "visualizations"

output_folder.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

df = df.dropna(subset=["Date", "AQI"])
df = df.sort_values("Date")

sns.set_theme(style="whitegrid")

# 3. AQI trend over time
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["AQI"], linewidth=1)
plt.title("Daily Air Quality Index Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_folder / "aqi_trend.png")
plt.show()
plt.close()

# 4. AQI category distribution
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="AQI_Category",
    order=[
        "Good", "Satisfactory", "Moderate",
        "Poor", "Very Poor", "Severe"
    ]
)
plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Number of Days")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_folder / "aqi_category_distribution.png")
plt.show()
plt.close()

# 5. AQI distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["AQI"], bins=30, kde=True)
plt.title("Distribution of Daily AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_folder / "aqi_distribution.png")
plt.show()
plt.close()

# 6. Monthly average AQI
df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly_aqi = df.groupby("Month")["AQI"].mean().reset_index()

plt.figure(figsize=(13, 5))
sns.lineplot(data=monthly_aqi, x="Month", y="AQI", marker="o")
plt.title("Monthly Average AQI")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(output_folder / "monthly_average_aqi.png")
plt.show()
plt.close()

# 7. Pollutant correlation heatmap
pollutant_columns = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "NH3 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)"
]

available_columns = [
    col for col in pollutant_columns if col in df.columns
]

if len(available_columns) >= 2:
    plt.figure(figsize=(10, 7))
    correlation = df[available_columns].corr()
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Pollutant Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_folder / "pollutant_correlation.png")
    plt.show()
    plt.close()

print("\nDATA VISUALIZATION COMPLETED")
print("Number of AQI records:", len(df))
print("Visualizations saved in:", output_folder)
print("\nGenerated files:")
for file in output_folder.glob("*.png"):
    print(file.name)