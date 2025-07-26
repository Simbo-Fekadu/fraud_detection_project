import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import time

# Start timer
start_time = time.time()

# Load datasets
try:
    fraud_data = pd.read_csv('Fraud_Data_transformed.csv')
    fraud_labels = pd.read_csv('Fraud_Data_labels.csv')
    creditcard_data = pd.read_csv('creditcard_scaled.csv')
    creditcard_labels = pd.read_csv('creditcard_labels.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Subsample for speed
sample_size = 1000  # Adjust based on your hardware
fraud_data = fraud_data.sample(n=min(sample_size, len(fraud_data)), random_state=42)
fraud_labels = fraud_labels.loc[fraud_data.index]
creditcard_data = creditcard_data.sample(n=min(sample_size, len(creditcard_data)), random_state=42)
creditcard_labels = creditcard_labels.loc[creditcard_data.index]

# Prepare Fraud_Data
min_len = min(len(fraud_data), len(fraud_labels))
X_fraud = fraud_data.iloc[:min_len].reset_index(drop=True)
y_fraud = fraud_labels.iloc[:min_len, 0].reset_index(drop=True)
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# Prepare creditcard
X_credit = creditcard_data
y_credit = creditcard_labels.iloc[:, 0]
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit)

# Train XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=3, n_jobs=-1)
xgb_model.fit(X_fraud_train, y_fraud_train)
xgb_model_credit = xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=3, n_jobs=-1)
xgb_model_credit.fit(X_credit_train, y_credit_train)

# SHAP explainer
explainer_fraud = shap.TreeExplainer(xgb_model)
shap_values_fraud = explainer_fraud.shap_values(X_fraud_test)

explainer_credit = shap.TreeExplainer(xgb_model_credit)
shap_values_credit = explainer_credit.shap_values(X_credit_test)

# Plot SHAP summary for Fraud_Data
plt.figure()
shap.summary_plot(shap_values_fraud, X_fraud_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Fraud_Data")
plt.tight_layout()
plt.savefig('shap_fraud_summary.png')
plt.close()

# Plot SHAP summary for creditcard
plt.figure()
shap.summary_plot(shap_values_credit, X_credit_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - creditcard")
plt.tight_layout()
plt.savefig('shap_creditcard_summary.png')
plt.close()

print("SHAP analysis complete. Plots saved as 'shap_fraud_summary.png' and 'shap_creditcard_summary.png'.")
print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")