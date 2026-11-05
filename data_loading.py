import pandas as pd
from pathlib import Path

# Locate the project folders
project_folder = Path(__file__).resolve().parent
raw_folder = project_folder / "data" / "raw"
output_folder = project_folder / "data"

# Find all CSV files
csv_files = list(raw_folder.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the raw folder.")
else:
    dataframes = []

    for file in csv_files:
        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file)
        df["source_file"] = file.name

        print("Shape:", df.shape)
        print("Columns:", list(df.columns))

        dataframes.append(df)

    # Combine all files
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Save combined dataset
    output_file = output_folder / "combined_air_quality_data.csv"
    combined_data.to_csv(output_file, index=False)

    print("\nAll files combined successfully!")
    print("Combined dataset shape:", combined_data.shape)
    print("Saved to:", output_file)
    print("\nFirst five rows:")
    print(combined_data.head())