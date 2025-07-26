import pandas as pd

# Load datasets
fraud_data = pd.read_csv('./Data/Fraud_Data.csv')
creditcard_data = pd.read_csv('./Data/creditcard.csv')

# Check for missing values
print("Fraud Data Missing Values:\n", fraud_data.isnull().sum())
print("\nCredit Card Data Missing Values:\n", creditcard_data.isnull().sum())

# Handle missing values in Fraud_Data.csv

# Example: Impute numerical columns with median, categorical with mode
for col in fraud_data.columns:
    if fraud_data[col].dtype in ['int64', 'float64']:
        fraud_data[col].fillna(fraud_data[col].median(), inplace=True)
    else:
        fraud_data[col].fillna(fraud_data[col].mode()[0], inplace=True)

# Handle missing values in creditcard.csv
# Since creditcard.csv typically has no missing values, confirm and proceed
if creditcard_data.isnull().sum().sum() == 0:
    print("No missing values in creditcard.csv")
else:
    for col in creditcard_data.columns:
        if creditcard_data[col].dtype in ['int64', 'float64']:
            creditcard_data[col].fillna(creditcard_data[col].median(), inplace=True)

# Save cleaned datasets
fraud_data.to_csv('Fraud_Data_cleaned.csv', index=False)
creditcard_data.to_csv('creditcard_cleaned.csv', index=False)
print("Cleaned datasets saved.")
