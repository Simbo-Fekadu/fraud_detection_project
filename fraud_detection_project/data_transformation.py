import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load datasets
fraud_data = pd.read_csv('Fraud_Data_features.csv')
creditcard_data = pd.read_csv('creditcard_cleaned.csv')

from sklearn.impute import SimpleImputer
# Handle class imbalance for Fraud_Data.csv
print("Fraud Data Class Distribution:\n", fraud_data['class'].value_counts())
X_fraud = fraud_data.drop(['class', 'signup_time', 'purchase_time', 'user_id', 'device_id'], axis=1)  # Drop non-feature columns
y_fraud = fraud_data['class']
# Impute missing values in numerical features
numerical_features = X_fraud.select_dtypes(include=['int64', 'float64'])
imputer = SimpleImputer(strategy='mean')
X_fraud_num_imputed = pd.DataFrame(imputer.fit_transform(numerical_features), columns=numerical_features.columns)
smote = SMOTE(random_state=42)
X_fraud_resampled, y_fraud_resampled = smote.fit_resample(X_fraud_num_imputed, y_fraud)
print("Resampled Fraud Data Class Distribution:\n", pd.Series(y_fraud_resampled).value_counts())

# Handle class imbalance for creditcard.csv
print("\nCredit Card Data Class Distribution:\n", creditcard_data['Class'].value_counts())
X_credit = creditcard_data.drop('Class', axis=1)
y_credit = creditcard_data['Class']
X_credit_resampled, y_credit_resampled = smote.fit_resample(X_credit, y_credit)
print("Resampled Credit Card Data Class Distribution:\n", pd.Series(y_credit_resampled).value_counts())

# Normalization and scaling
scaler = StandardScaler()
X_fraud_scaled = scaler.fit_transform(X_fraud_resampled)
X_credit_scaled = scaler.fit_transform(X_credit_resampled)

# Encode categorical features for Fraud_Data.csv
categorical_cols = ['source', 'browser', 'sex', 'country']
numerical_cols = X_fraud.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])
X_fraud_transformed = preprocessor.fit_transform(X_fraud)

# Save transformed datasets
pd.DataFrame(X_fraud_scaled).to_csv('Fraud_Data_scaled.csv', index=False)
pd.Series(y_fraud_resampled).to_csv('Fraud_Data_labels.csv', index=False)
pd.DataFrame(X_fraud_transformed).to_csv('Fraud_Data_transformed.csv', index=False)
pd.DataFrame(X_credit_scaled).to_csv('creditcard_scaled.csv', index=False)
pd.Series(y_credit_resampled).to_csv('creditcard_labels.csv', index=False)
print("Data transformation complete. Saved transformed datasets.")