import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import time

start_time = time.time()

# Load datasets
try:
    fraud_data = pd.read_csv('Fraud_Data_features.csv')
    creditcard_data = pd.read_csv('creditcard_cleaned.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Handle NaN in Fraud_Data
print("Checking for NaN in Fraud_Data:\n", fraud_data.isna().sum())
numerical_cols = ['purchase_value', 'age', 'ip_address', 'transaction_frequency', 'transaction_velocity', 'hour_of_day', 'day_of_week', 'time_since_signup']
imputer = SimpleImputer(strategy='median')
fraud_data[numerical_cols] = imputer.fit_transform(fraud_data[numerical_cols])

# Prepare Fraud_Data
print("\nFraud Data Class Distribution:\n", fraud_data['class'].value_counts())
X_fraud = fraud_data.drop(['class', 'signup_time', 'purchase_time', 'user_id', 'device_id'], axis=1)
y_fraud = fraud_data['class']

# Encode categorical features
categorical_cols = ['source', 'browser', 'sex', 'country']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
X_fraud_transformed = preprocessor.fit_transform(X_fraud)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_fraud_resampled, y_fraud_resampled = smote.fit_resample(X_fraud_transformed, y_fraud)
print("Resampled Fraud Data Class Distribution:\n", pd.Series(y_fraud_resampled).value_counts())

# Scale numerical features
scaler = StandardScaler()
X_fraud_scaled = scaler.fit_transform(X_fraud_resampled)

# Prepare creditcard
print("\nCredit Card Data Class Distribution:\n", creditcard_data['Class'].value_counts())
X_credit = creditcard_data.drop('Class', axis=1)
y_credit = creditcard_data['Class']
X_credit_resampled, y_credit_resampled = smote.fit_resample(X_credit, y_credit)
print("Resampled Credit Card Data Class Distribution:\n", pd.Series(y_credit_resampled).value_counts())

# Scale creditcard data
X_credit_scaled = scaler.fit_transform(X_credit_resampled)

# Save transformed datasets
pd.DataFrame(X_fraud_scaled, columns=[f'feature_{i}' for i in range(X_fraud_scaled.shape[1])]).to_csv('Fraud_Data_transformed.csv', index=False)
pd.Series(y_fraud_resampled).to_csv('Fraud_Data_labels.csv', index=False)
pd.DataFrame(X_credit_scaled, columns=X_credit.columns).to_csv('creditcard_scaled.csv', index=False)
pd.Series(y_credit_resampled).to_csv('creditcard_labels.csv', index=False)
print(f"Data transformation complete. Execution time: {(time.time() - start_time) / 60:.2f} minutes")