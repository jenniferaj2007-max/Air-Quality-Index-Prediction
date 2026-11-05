# Air Quality Index Prediction

## 1. Project Overview
This project estimates the Air Quality Index (AQI) using air pollutant measurements collected from the Central Pollution Control Board (CPCB).

## 2. Dataset
- Source: CPCB
- Location: Anand Vihar, Delhi
- Period: 2024–2026
- Original records: 24,840
- Daily AQI records: 948 days with calculated AQI
- Input features: 7 pollutant measurements

## 3. Algorithms Used
1. Linear Regression
2. Random Forest Regression

## 4. Model Performance

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 28.18 | 3.59 |
| RMSE | 41.31 | 8.77 |
| R2 Score | 0.9151 | 0.9962 |

## 5. Best Model
Random Forest Regression was selected based on its lower Mean Absolute Error.

## 6. Visualizations
- AQI trend
- AQI category distribution
- AQI distribution
- Monthly average AQI
- Pollutant correlation
- Model comparison
- Feature importance

## 7. Project Structure
- data: Raw and processed datasets
- models: Trained machine learning models
- outputs: Model predictions and comparison
- visualizations: Generated charts
- Python files: Data processing, training and prediction

## 8. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 9. Disclaimer
This project demonstrates educational AQI estimation using pollutant measurements. It is not an official CPCB AQI service or a future-day forecasting system.