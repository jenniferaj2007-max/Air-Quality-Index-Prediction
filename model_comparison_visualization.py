import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set paths
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "outputs" / "model_comparison.csv"
output_folder = project_folder / "visualizations"

output_folder.mkdir(parents=True, exist_ok=True)

# 2. Load comparison results
df = pd.read_csv(input_file)

metrics = ["MAE", "RMSE", "R2 Score"]

# 3. Create comparison charts
for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Machine Learning Model")
    plt.ylabel(metric)

    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + "_comparison.png"

    plt.savefig(output_folder / filename)
    plt.show()
    plt.close()

print("\nMODEL COMPARISON VISUALIZATION COMPLETED")
print("Saved in:", output_folder)
print("\nGenerated charts:")
for file in output_folder.glob("*_comparison.png"):
    print(file.name)