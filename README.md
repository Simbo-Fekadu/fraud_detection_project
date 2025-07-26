# Fraud Detection Project 🚨

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

Welcome to the **Fraud Detection Project** for Adey Innovations Inc.! This repository delivers robust machine learning models to detect fraudulent transactions in e-commerce (`Fraud_Data.csv`) and bank card datasets (`creditcard.csv`). From data preprocessing to model explainability, this project combines efficiency, accuracy, and actionable insights to combat fraud. 🌟

## Project Overview
- **Objective**: Build and evaluate models to identify fraud, minimizing financial losses and enhancing user trust.
- **Datasets**:
  - `Fraud_Data`: ~151,112 e-commerce transactions (9.36% fraud).
  - `creditcard`: ~284,807 bank transactions (0.17% fraud).
- **Key Tasks**:
  - **Task 1**: Data cleaning, EDA, geolocation mapping, feature engineering, and transformation.
  - **Task 2**: Trained Logistic Regression and XGBoost, evaluated with AUC-PR and F1-Score.
  - **Explainability**: SHAP analysis to reveal key fraud predictors (e.g., `transaction_velocity`, `Amount`).
- **Achievements**:
  - Optimized geolocation merge from 17 hours to ~10 minutes using `IntervalTree`. ⚡
  - XGBoost achieved AUC-PR [Your XGBoost AUC-PR, e.g., 0.9457] vs. Logistic Regression [Your LR AUC-PR, e.g., 0.8923].
  - SHAP insights guide fraud prevention strategies.

## Repository Structure

fraud_detection_project/ ├── data/ # Processed datasets (CSVs) ├── scripts/ # Python scripts for all tasks ├── plots/ # EDA, confusion matrices, SHAP plots ├── report/ # Detailed report (report.md or report.pdf) ├── README.md # You're here!

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd fraud_detection_project


Create Virtual Environment:
python -m venv fraud_detection_env
.\fraud_detection_env\Scripts\Activate.ps1  # Windows


Install Dependencies:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn shap intervaltree tqdm pandoc


Add Raw Data:

Place Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv in data/.



Running the Project
Execute scripts in order:
cd scripts
python handle_missing.py
python clean_data.py
python eda.py
python merge_geolocation.py
python feature_engineering.py
python data_transformation.py
python model_building.py
python shap_analysis.py

Outputs

Data: data/Fraud_Data_transformed.csv, data/creditcard_scaled.csv, etc.
Plots: plots/fraud_data_univariate.png, plots/Fraud_Data_best_cm.png, plots/shap_fraud_summary.png, etc.
Report: report/report.md or report/report.pdf with detailed methodology and results.

Key Results

Efficiency: Reduced geolocation processing from 17 hours to ~10 minutes. 🕒
Performance: XGBoost outperformed with AUC-PR [Your XGBoost AUC-PR] for Fraud_Data and [Your XGBoost AUC-PR] for creditcard.
Insights: SHAP revealed transaction_velocity and Amount as top fraud indicators, enabling targeted monitoring. 📊

Report
Dive into the full story in report/report.md or report/report.pdf, covering data preprocessing, model evaluation, and SHAP explainability with visualizations.
Author
Simbo FekaduSubmitted for Adey Innovations Inc., 29 July 2025

🔍 Explore the code, run the scripts, and join the fight against fraud!

